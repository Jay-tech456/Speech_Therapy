import sys
from typing import Dict, Any
import myprosody

class MyProsody_Tool():
    def __init__(self):
        self.analyze = self._analyze

    def _analyze(self, audio_path: str) -> Dict[str, Any]:
        """
        Analyze audio using MyProsody (requires .wav)
        """

        try:
            # MyProsody requires a filename *without extension*
            # and that file must be placed in myprosody/MyProsody folder
            
            file_id = audio_path.split("/")[-1].replace(".wav", "")

            # Extract metrics using MyProsody's mysp class
            results = {
                "duration": mysp.myspgend(audio_path),
                "syllables": mysp.myspsyl(audio_path),
                "pauses": mysp.mysppaus(audio_path),
                "rate_of_speech": mysp.myspsr(audio_path),
                "articulation_rate": mysp.myspatc(audio_path),
                "f0_mean": mysp.myspf0mean(audio_path),
                "f0_std": mysp.myspf0sd(audio_path),
                "f0_median": mysp.myspf0med(audio_path),
                "speaking_style": mysp.myspst(audio_path),
            }

            return {
                "message": "success",
                "analysis": results
            }

        except Exception as e:
            return {"message": f"Error: {str(e)}"}
