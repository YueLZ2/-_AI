from train import Trainer


trainer = Trainer(
    data_path='./DATA/smartmedicine/processed/range_wavelet.npy',
    moder_w_path='result/range_wavelet/best_model.pth',
    log_path='result/range_wavelet/training_log.txt',
    learning_rate=1e-6,
    num_epochs=50,
    batch_size=32,
    patience=30
)

trainer.train()
# trainer.test()