from transformers import AutoProcessor, BarkModel
import scipy.io

processor = AutoProcessor.from_pretrained("suno/bark")
model = BarkModel.from_pretrained("suno/bark")

voice_preset = "v2/en_speaker_6"

inputs = processor("Hello, my dog is cute", voice_preset=voice_preset)

audio_array = model.generate(**inputs)
audio_array = audio_array.cpu().numpy().squeeze()


##save the audio file
sample_rate = model.generation_config.sample_rate
scipy.io.wavefile.write("audio1.wav",rate=sample_rate, data=audio_array)