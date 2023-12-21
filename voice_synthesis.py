from TTS.tts.configs.bark_config import BarkConfig
from TTS.tts.models.bark import Bark
import scipy

config = BarkConfig()
model = Bark.init_from_config(config)
model.load_checkpoint(config, checkpoint_dir="C:\\Users\\Jaswanth Kakani\\OneDrive\\Desktop\\saikumar\\sunobark\\suno\\bark",eval=True)


text= "My favorite food is pizza. It has cheese, tomatoes, pepperoni on top. I like the crispy crust. Whenever we have a party, we order pizza and everyone enjoys it"
output_dict = model.synthesize(text, config, speaker_id="speaker", voice_dirs="C:\\Users\\Jaswanth Kakani\\OneDrive\\Desktop\\saikumar\\sunobark\\bark_voices")

sample_rate = 16000

scipy.io.wavfile.write("bark_out.wav", rate=sample_rate, data=output_dict["wav"])