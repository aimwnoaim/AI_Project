CLASS_NAMES = ['plastic', 'metal', 'paper_cardboard', 'other']
CONFIDENCE_THRESHOLD = 0.80
SEED = 42
INPUT_SIZE = (224, 224)

def map_label(category):
    return {'plastic': 'plastic', 'metal': 'metal', 'paper': 'paper_cardboard',
            'cardboard': 'paper_cardboard'}.get(category.strip().lower(), 'other')
