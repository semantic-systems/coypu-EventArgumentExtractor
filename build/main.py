from flask import Flask, jsonify, request
from flask_healthz import healthz
from flair.data import Sentence
from flair.models import SequenceTagger


app = Flask(__name__)

app.register_blueprint(healthz, url_prefix="/healthz")


def liveness():
    pass


def readiness():
    pass


app.config.update(
    HEALTHZ = {
        "live": app.name + ".liveness",
        "ready": app.name + ".readiness"
    }
)

# load tagger
tagger = SequenceTagger.load("flair/ner-german")


@app.route('/', methods=['POST'])
def main():
    authenticated = False

    if 'key' in request.json:
        key = request.json['key']
        if (key == 'E1CVJ4RJKO0BLVFY'): authenticated = True

    if (authenticated == False):
        response = {'error': 'no valid API key'}
        http_code = 401

    elif ('message' in request.json):
        sentence = Sentence(str(request.json['message']))
        # predict NER tags
        tagger.predict(sentence)
        entities = {entity.text: {"tag": entity.tag, "confidence": entity.score} for entity in sentence.get_spans('ner')}
        response = {"sentence": str(request.json['message']),
                    'entities': entities}
        http_code = 200

    else:
        response = {'error': 'no valid input'}
        http_code = 400

    return jsonify(response), http_code


if __name__ == "__main__":
    app.run(host='0.0.0.0:5290')
