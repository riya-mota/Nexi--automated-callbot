from fastapi import FastAPI, HTTPException, File, UploadFile, Form
from fastapi.responses import FileResponse, StreamingResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import torch
import torchaudio
import torchaudio.transforms as T
import logging
import asyncio
import io
import base64
import os
import uuid
from pathlib import Path
from typing import Optional
import tempfile
import time

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize FastAPI app
app = FastAPI(
    title="F5-TTS Voice Cloning API",
    description="Real-time voice cloning using F5-TTS",
    version="1.0.0"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global variables for model caching
model = None
device = None
f5tts_api = None

# Pydantic models for request/response
class VoiceCloneRequest(BaseModel):
    text: str
    reference_text: Optional[str] = None
    speed: Optional[float] = 1.0
    cross_fade_duration: Optional[float] = 0.15
    return_format: Optional[str] = "wav"  # wav, mp3, base64

class VoiceCloneResponse(BaseModel):
    success: bool
    message: str
    audio_data: Optional[str] = None  # base64 encoded audio
    file_path: Optional[str] = None
    generation_time: Optional[float] = None

# Global reference audio settings
REFERENCE_AUDIO_PATH = "./reference.wav"
DEFAULT_REFERENCE_TEXT = "You just tell me what works, date, time, and your preferred location. We'll arrange everything for you. No pressure. Seriously. I'm just here to help you"

class VoiceCloner:
    def __init__(self):
        self.model = None
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.f5tts_api = None
        self.reference_audio = None
        self.reference_text = DEFAULT_REFERENCE_TEXT
        self.model_loaded = False
    
    async def load_model(self):
        """Load the F5-TTS model"""
        if self.model_loaded:
            return True
            
        try:
            logger.info("Loading F5-TTS model...")
            
            # Method 1: Try F5-TTS API first
            try:
                from f5_tts.api import F5TTS
                self.f5tts_api = F5TTS()
                self.model_loaded = True
                logger.info("✅ F5-TTS API loaded successfully")
                return True
            except Exception as api_error:
                logger.warning(f"API loading failed: {api_error}")
            
            # Method 2: Load model manually
            try:
                from f5_tts.model import DiT
                from f5_tts.infer.utils_infer import load_checkpoint
                
                model_cfg = {
                    'dim': 1024, 'depth': 22, 'heads': 16,
                    'ff_mult': 2, 'text_dim': 512, 'conv_layers': 4
                }
                
                self.model = DiT(**model_cfg)
                
                # Load from huggingface
                from huggingface_hub import hf_hub_download
                model_path = hf_hub_download(
                    repo_id="SWivid/F5-TTS",
                    filename="F5TTS_Base/model_1200000.safetensors"
                )
                
                from safetensors.torch import load_file
                state_dict = load_file(model_path)
                self.model.load_state_dict(state_dict, strict=False)
                self.model.to(self.device)
                self.model.eval()
                
                self.model_loaded = True
                logger.info("✅ Manual model loaded successfully")
                return True
                
            except Exception as manual_error:
                logger.error(f"Manual model loading failed: {manual_error}")
                return False
                
        except Exception as e:
            logger.error(f"Model loading failed: {e}")
            return False
    
    def preprocess_reference_audio(self, audio_path: str):
        """Preprocess reference audio"""
        try:
            if not Path(audio_path).exists():
                raise FileNotFoundError(f"Reference audio not found: {audio_path}")
            
            waveform, sample_rate = torchaudio.load(audio_path)
            
            # Convert to mono
            if waveform.shape[0] > 1:
                waveform = torch.mean(waveform, dim=0, keepdim=True)
            
            # Resample to 24kHz
            if sample_rate != 24000:
                resampler = T.Resample(sample_rate, 24000)
                waveform = resampler(waveform)
            
            self.reference_audio = waveform.to(self.device)
            return True
            
        except Exception as e:
            logger.error(f"Reference audio preprocessing failed: {e}")
            return False
    
    async def generate_voice(self, text: str, reference_text: str = None, speed: float = 1.0, cross_fade_duration: float = 0.15):
        """Generate cloned voice"""
        start_time = time.time()
        
        if not self.model_loaded:
            await self.load_model()
        
        if reference_text is None:
            reference_text = self.reference_text
        
        try:
            # Method 1: Use F5-TTS API
            if self.f5tts_api:
                audio = self.f5tts_api.infer(
                    ref_file=REFERENCE_AUDIO_PATH,
                    ref_text=reference_text,
                    gen_text=text,
                    cross_fade_duration=cross_fade_duration,
                    speed=speed
                )
                
                # Handle audio output
                if isinstance(audio, tuple):
                    audio_tensor = audio[0]
                    sample_rate = audio[1] if len(audio) > 1 else 24000
                else:
                    audio_tensor = audio
                    sample_rate = 24000
                
                # Convert to numpy
                if hasattr(audio_tensor, 'numpy'):
                    audio_np = audio_tensor.numpy()
                elif hasattr(audio_tensor, 'detach'):
                    audio_np = audio_tensor.detach().cpu().numpy()
                else:
                    audio_np = audio_tensor
                
                # Ensure correct shape
                if len(audio_np.shape) > 1:
                    if audio_np.shape[0] > audio_np.shape[1]:
                        audio_np = audio_np[0]
                    else:
                        audio_np = audio_np.flatten()
                
                generation_time = time.time() - start_time
                return audio_np, sample_rate, generation_time
            
            # Method 2: Manual inference
            elif self.model:
                if self.reference_audio is None:
                    if not self.preprocess_reference_audio(REFERENCE_AUDIO_PATH):
                        raise Exception("Failed to preprocess reference audio")
                
                from f5_tts.infer.utils_infer import infer_process
                
                try:
                    final_wave, final_sample_rate, _ = infer_process(
                        ref_audio=self.reference_audio,
                        ref_text=reference_text,
                        gen_text=text,
                        model=self.model,
                        cross_fade_duration=cross_fade_duration,
                        speed=speed
                    )
                except TypeError:
                    final_wave, final_sample_rate, _ = infer_process(
                        self.reference_audio, reference_text, text, self.model
                    )
                
                # Handle output format
                if isinstance(final_wave, tuple):
                    final_wave = final_wave[0]
                
                if len(final_wave.shape) == 1:
                    final_wave = final_wave.unsqueeze(0)
                elif len(final_wave.shape) == 3:
                    final_wave = final_wave.squeeze(0)
                
                audio_np = final_wave.cpu().numpy()
                if len(audio_np.shape) > 1:
                    audio_np = audio_np[0]
                
                generation_time = time.time() - start_time
                return audio_np, final_sample_rate, generation_time
            
            else:
                raise Exception("No model available")
                
        except Exception as e:
            logger.error(f"Voice generation failed: {e}")
            raise

# Initialize voice cloner
voice_cloner = VoiceCloner()

@app.on_event("startup")
async def startup_event():
    """Initialize the model on startup"""
    logger.info("🚀 Starting F5-TTS Voice Cloning API...")
    await voice_cloner.load_model()
    logger.info("✅ API ready for voice cloning!")

@app.get("/")
async def root():
    """Root endpoint"""
    return {
        "message": "F5-TTS Voice Cloning API",
        "status": "running",
        "model_loaded": voice_cloner.model_loaded,
        "device": voice_cloner.device
    }

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "model_loaded": voice_cloner.model_loaded,
        "device": voice_cloner.device,
        "reference_audio_exists": Path(REFERENCE_AUDIO_PATH).exists()
    }

@app.post("/upload-reference", response_model=dict)
async def upload_reference_audio(
    audio_file: UploadFile = File(...),
    reference_text: str = Form(...)
):
    """Upload reference audio file"""
    try:
        # Save uploaded file
        with open(REFERENCE_AUDIO_PATH, "wb") as f:
            content = await audio_file.read()
            f.write(content)
        
        # Update reference text
        voice_cloner.reference_text = reference_text
        voice_cloner.reference_audio = None  # Reset to force reprocessing
        
        return {
            "success": True,
            "message": "Reference audio uploaded successfully",
            "filename": audio_file.filename,
            "reference_text": reference_text
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to upload reference audio: {str(e)}")

@app.post("/clone-voice", response_model=VoiceCloneResponse)
async def clone_voice(request: VoiceCloneRequest):
    """Clone voice with given text"""
    try:
        if not Path(REFERENCE_AUDIO_PATH).exists():
            raise HTTPException(
                status_code=400, 
                detail="Reference audio not found. Please upload reference audio first."
            )
        
        # Generate voice
        audio_data, sample_rate, generation_time = await voice_cloner.generate_voice(
            text=request.text,
            reference_text=request.reference_text,
            speed=request.speed,
            cross_fade_duration=request.cross_fade_duration
        )
        
        # Save to temporary file
        output_filename = f"generated_{uuid.uuid4().hex}.wav"
        output_path = f"./temp/{output_filename}"
        
        # Ensure temp directory exists
        os.makedirs("./temp", exist_ok=True)
        
        # Save audio file
        import soundfile as sf
        sf.write(output_path, audio_data, sample_rate)
        
        response_data = {
            "success": True,
            "message": "Voice generated successfully",
            "file_path": output_path,
            "generation_time": generation_time
        }
        
        # Handle different return formats
        if request.return_format == "base64":
            # Convert to base64
            audio_bytes = io.BytesIO()
            sf.write(audio_bytes, audio_data, sample_rate, format='WAV')
            audio_bytes.seek(0)
            audio_b64 = base64.b64encode(audio_bytes.read()).decode()
            response_data["audio_data"] = audio_b64
        
        return VoiceCloneResponse(**response_data)
        
    except Exception as e:
        logger.error(f"Voice cloning failed: {e}")
        raise HTTPException(status_code=500, detail=f"Voice cloning failed: {str(e)}")

@app.post("/clone-voice-stream")
async def clone_voice_stream(request: VoiceCloneRequest):
    """Clone voice and return as streaming audio"""
    try:
        if not Path(REFERENCE_AUDIO_PATH).exists():
            raise HTTPException(
                status_code=400, 
                detail="Reference audio not found. Please upload reference audio first."
            )
        
        # Generate voice
        audio_data, sample_rate, _ = await voice_cloner.generate_voice(
            text=request.text,
            reference_text=request.reference_text,
            speed=request.speed,
            cross_fade_duration=request.cross_fade_duration
        )
        
        # Convert to bytes
        audio_bytes = io.BytesIO()
        import soundfile as sf
        sf.write(audio_bytes, audio_data, sample_rate, format='WAV')
        audio_bytes.seek(0)
        
        return StreamingResponse(
            io.BytesIO(audio_bytes.read()),
            media_type="audio/wav",
            headers={"Content-Disposition": "attachment; filename=generated_voice.wav"}
        )
        
    except Exception as e:
        logger.error(f"Voice cloning stream failed: {e}")
        raise HTTPException(status_code=500, detail=f"Voice cloning failed: {str(e)}")

@app.get("/download/{filename}")
async def download_audio(filename: str):
    """Download generated audio file"""
    file_path = f"./temp/{filename}"
    if not Path(file_path).exists():
        raise HTTPException(status_code=404, detail="File not found")
    
    return FileResponse(
        file_path,
        media_type="audio/wav",
        filename=filename
    )

@app.delete("/cleanup")
async def cleanup_temp_files():
    """Clean up temporary audio files"""
    try:
        temp_dir = Path("./temp")
        if temp_dir.exists():
            for file in temp_dir.glob("generated_*.wav"):
                file.unlink()
        
        return {"success": True, "message": "Temporary files cleaned up"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Cleanup failed: {str(e)}")

# Batch processing endpoint
@app.post("/clone-voice-batch")
async def clone_voice_batch(texts: list[str], reference_text: str = None, speed: float = 1.0):
    """Clone multiple texts in batch"""
    try:
        if not Path(REFERENCE_AUDIO_PATH).exists():
            raise HTTPException(
                status_code=400, 
                detail="Reference audio not found. Please upload reference audio first."
            )
        
        results = []
        
        for i, text in enumerate(texts):
            try:
                audio_data, sample_rate, generation_time = await voice_cloner.generate_voice(
                    text=text,
                    reference_text=reference_text,
                    speed=speed
                )
                
                # Save to file
                output_filename = f"batch_{i}_{uuid.uuid4().hex}.wav"
                output_path = f"./temp/{output_filename}"
                
                import soundfile as sf
                sf.write(output_path, audio_data, sample_rate)
                
                results.append({
                    "index": i,
                    "text": text,
                    "success": True,
                    "file_path": output_path,
                    "generation_time": generation_time
                })
                
            except Exception as e:
                results.append({
                    "index": i,
                    "text": text,
                    "success": False,
                    "error": str(e)
                })
        
        return {
            "success": True,
            "message": f"Batch processing completed. {len([r for r in results if r['success']])} successful.",
            "results": results
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Batch processing failed: {str(e)}")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)