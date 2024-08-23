import logging
from django.shortcuts import render
from transformers import pipeline

logger = logging.getLogger(__name__)

model_name = "distilbert-base-uncased-finetuned-sst-2-english"
classifier = pipeline("sentiment-analysis", model=model_name, device=0)

def analyze_sentiment(request):
    logger.debug('Função analyze_sentiment chamada.')
    if request.method == 'POST':
        text = request.POST.get('text', '')
        logger.debug(f'Texto recebido: {text}')
        if not text:
            logger.error('Texto vazio')
            return render(request, 'sentiment/index.html', {'error': 'Texto não pode estar vazio'})
        
        try:
            result = classifier(text)
            sentiment = result[0]['label']
            score = result[0]['score']
            logger.debug(f'Resultado da análise: {result}')
        except Exception as e:
            logger.error(f'Erro na análise de sentimento: {str(e)}')
            return render(request, 'sentiment/index.html', {'error': str(e)})
        
        return render(request, 'sentiment/index.html', {'text': text, 'sentiment': sentiment, 'score': score})
    
    return render(request, 'sentiment/index.html')
