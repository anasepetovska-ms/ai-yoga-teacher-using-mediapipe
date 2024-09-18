import azure.cognitiveservices.speech as speechsdk 

# Replace with your own subscription key and service region (e.g., "westus").
# Get from config
speech_key = ""
region = ""

def synthesize_speech_audio(text):
    # Configure the Azure Text to Speech instance
    speech_config = speechsdk.SpeechConfig(subscription=speech_key, region=region)
    
    speechSynthesisVoiceName  = "en-US-AvaMultilingualNeural"; 
    # "en-US-AriaNeural"
    speech_config.speech_synthesis_voice_name = speechSynthesisVoiceName
    # audio_config = speechsdk.audio.AudioOutputConfig(use_default_speaker=True)
    speech_synthesizer = speechsdk.SpeechSynthesizer(speech_config=speech_config, audio_config=None)

    # Synthesize text to speech
    result = speech_synthesizer.speak_text_async(text).get()
    if result.reason == speechsdk.ResultReason.SynthesizingAudioCompleted:
        print("Speech synthesized to speaker for text [{}]".format(text))
    elif result.reason == speechsdk.ResultReason.Canceled:
        cancellation_details = result.cancellation_details
        print("Speech synthesis canceled: {}".format(cancellation_details.reason))
        if cancellation_details.reason == speechsdk.CancellationReason.Error:
            if cancellation_details.error_details:
                print("Error details: {}".format(cancellation_details.error_details))
        print("Did you update the subscription info?")
    wav_data = result.audio_data

    return wav_data