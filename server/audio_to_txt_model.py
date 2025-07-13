from transformers import WhisperProcessor, WhisperForConditionalGeneration
import torchaudio
import torchaudio.transforms as T


def hugf_aud_to_txt_Model(audio):

    processor = WhisperProcessor.from_pretrained("openai/whisper-tiny")
    model = WhisperForConditionalGeneration.from_pretrained("openai/whisper-tiny")
    model.config.forced_decoder_ids = None

    # Load your own MP3 file
    waveform, sample_rate = torchaudio.load(audio)  # waveform is a tensor

    if waveform.shape[0] > 1:
        waveform = waveform.mean(dim=0, keepdim=True)  # Average the channels


    resampler = T.Resample(orig_freq=sample_rate, new_freq=16000)
    waveform = resampler(waveform)


    # Now squeeze to remove channel dimension (safe now!)
    waveform = waveform.squeeze(0)  # (samples,)

    # Now feed to processor
    input_features = processor(waveform, sampling_rate=16000, return_attention_mask=True, return_tensors="pt")

    # Model inference
    predicted_ids = model.generate(input_features["input_features"],attention_mask=input_features["attention_mask"], max_new_tokens=440,language='bn')

    # Decode token ids to text
    transcription = processor.batch_decode(predicted_ids, skip_special_tokens=True)

    return transcription[0]

#print(hugf_aud_to_txt_Model())

