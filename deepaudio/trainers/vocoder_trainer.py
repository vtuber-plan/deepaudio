from deepaudio.trainers.train import train

def vocoder_pretrain():
    train(create_pretrain_dataset, PretrainLanguageModel)

if __name__ == "__main__":
    vocoder_pretrain()