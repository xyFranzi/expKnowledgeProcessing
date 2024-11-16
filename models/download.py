import fasttext.util

# run this file ro download model 'cc.en.300.bin'

fasttext.util.download_model('en', if_exists='ignore')  # English
# ft = fasttext.load_model('cc.en.300.bin')