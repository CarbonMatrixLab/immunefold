from abfold.data.abrep import AbRepExtractor


if __name__ == '__main__':
    ab = AbRepExtractor()
    model_path = '../ab-external/abrep/bert_epoch12_step480000'
    print(ab.get(model_path))
