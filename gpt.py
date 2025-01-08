from fastapi import FastAPI, Request, HTTPException
from fastapi.responses import HTMLResponse
from pydantic import BaseModel
from transformers import pipeline, DistilBertTokenizerFast, DistilBertForSequenceClassification
from fastapi.templating import Jinja2Templates

# FastAPI 애플리케이션 생성 
app = FastAPI()

# Jinja2 템플릿 디렉터리 설정
templates = Jinja2Templates(directory="/Users/sunhacho/Downloads/Image and Source/templates")

# 모델과 토크나이저 로드
model = DistilBertForSequenceClassification.from_pretrained("/Users/sunhacho/Downloads/saved_model3")
tokenizer = DistilBertTokenizerFast.from_pretrained("/Users/sunhacho/Downloads/saved_model3")

# 예측 파이프라인 생성
classifier = pipeline("text-classification", model=model, tokenizer=tokenizer)

class TextRequest(BaseModel):
    text: str

@app.get("/", response_class=HTMLResponse)
async def read_root(request: Request):
    return templates.TemplateResponse("index2.html", {"request": request})

@app.get("/predict", response_class=HTMLResponse)
async def show_predict_page(request: Request):
    return templates.TemplateResponse("main.html", {"request": request})

from collections import Counter

def predict_paragraph(paragraph, classifier):
    sentences = paragraph.split('. ')
    predictions = []
    scores = []
    
    for sentence in sentences:
        prediction = classifier(sentence)
        label = prediction[0]['label']
        score = prediction[0]['score']
        
        predictions.append(label)
        scores.append(score)
    
    # 최빈값으로 최종 결과 결정
    most_common_label = Counter(predictions).most_common(1)[0][0]
    
    # 평균 점수를 참고로 추가 반환
    avg_score = sum(scores) / len(scores)
    
    return most_common_label, avg_score

@app.post("/predict/")
async def predict_text(request: TextRequest):
    input_text = request.text
    
    # 예측 실행 (문단 단위 처리)
    label, avg_score = predict_paragraph(input_text, classifier)
    
    # 결과 반환
    if label == 'LABEL_0':  # GPT 생성 텍스트가 아닌 경우
        return {"message": f"입력 텍스트는 GPT 생성 텍스트가 아닐 가능성이 높습니다.", "score": avg_score}
    else:
        return {"message": f"입력 텍스트는 GPT 생성 텍스트일 가능성이 높습니다.", "score": avg_score}

