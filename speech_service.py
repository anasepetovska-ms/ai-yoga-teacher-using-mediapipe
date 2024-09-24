import azure.cognitiveservices.speech as speechsdk 

# Replace with your own subscription key and service region (e.g., "westus").
# Get from config
speech_key = ""
region = ""

def speech_synthesis_to_audio_data_stream(text):
    """performs speech synthesis and gets the audio data from single request based stream."""
    # Creates an instance of a speech config with specified subscription key and service region.
    speech_config = speechsdk.SpeechConfig(subscription=speech_key, region=region)
    # Creates a speech synthesizer with a null output stream.
    # This means the audio output data will not be written to any output channel.
    # You can just get the audio from the result.
    speech_synthesizer = speechsdk.SpeechSynthesizer(speech_config=speech_config, audio_config=None)

    result = speech_synthesizer.speak_text_async(text).get()
    # Check result
    if result.reason == speechsdk.ResultReason.SynthesizingAudioCompleted:
        print("Speech synthesized for text [{}]".format(text))
        audio_data_stream = speechsdk.AudioDataStream(result)

        # You can save all the data in the audio data stream to a file
        file_name = "outputaudio.wav"
        audio_data_stream.save_to_wav_file(file_name)
        print("Audio data for text [{}] was saved to [{}]".format(text, file_name))

        # You can also read data from audio data stream and process it in memory
        # Reset the stream position to the beginning since saving to file puts the position to end.
        audio_data_stream.position = 0

        # Reads data from the stream
        audio_buffer = bytes(16000)
        total_size = 0
        filled_size = audio_data_stream.read_data(audio_buffer)
        while filled_size > 0:
            print("{} bytes received.".format(filled_size))
            total_size += filled_size
            filled_size = audio_data_stream.read_data(audio_buffer)
        print("Totally {} bytes received for text [{}].".format(total_size, text))
    elif result.reason == speechsdk.ResultReason.Canceled:
        cancellation_details = result.cancellation_details
        print("Speech synthesis canceled: {}".format(cancellation_details.reason))
        if cancellation_details.reason == speechsdk.CancellationReason.Error:
            print("Error details: {}".format(cancellation_details.error_details))

    return result.audio_data

def synthesize_speech_audio(text):
    # Configure the Azure Text to Speech instance
    speech_config = speechsdk.SpeechConfig(subscription=speech_key, region=region)
    
    speechSynthesisVoiceName  = "en-US-AriaNeural" 
    # "en-US-AvaMultilingualNeural"; 
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