from flask import Flask, request ,jsonify
from gevent import pywsgi
import json

from modelscope.pipelines import pipeline
from modelscope.utils.constant import Tasks
pipeline = pipeline(Tasks.faq_question_answering, 'damo/nlp_faq-question-answering_multilingual-base')

from modelscope import AutoModelForCausalLM, AutoTokenizer
from modelscope import GenerationConfig
tokenizer = AutoTokenizer.from_pretrained("qwen/Qwen-7B-Chat", revision = 'v1.0.1',trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained("qwen/Qwen-7B-Chat", revision = 'v1.0.1',device_map="auto", trust_remote_code=True).eval()
model.generation_config = GenerationConfig.from_pretrained("Qwen/Qwen-7B-Chat",revision = 'v1.0.1', trust_remote_code=True) # 可指定不同的生成长度、top_p等相关超参


app = Flask(__name__)
@app.route('/nlp_faq', methods=['GET', 'POST'])
def nlp_faq():
    if request.method == 'GET':
        input = request.args.get('input')
    else:
         input = request.json.get('input')
    
    print(input)
    output = pipeline(input)
    return output

@app.route('/qwen', methods=['get','post'])
def qwen():
    if request.method == 'GET':
        input = request.args.get('input')
    else:
        input = request.json.get('input')
    response, history = model.chat(tokenizer,input, history=None)
    print(response)
    return response


server = pywsgi.WSGIServer(('127.0.0.1', 5001), app)
server.serve_forever()
