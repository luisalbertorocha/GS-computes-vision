FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 PIP_NO_CACHE_DIR=1
WORKDIR /app

RUN apt-get update \
    && apt-get install -y --no-install-recommends libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install -r requirements.txt

COPY src ./src
COPY scheduler ./scheduler
COPY weights ./weights
COPY samples ./samples
COPY .streamlit ./.streamlit

RUN useradd --create-home --uid 1000 app && mkdir -p /app/data && chown -R app /app
USER app

EXPOSE 8000 8501
# Padrão da imagem = deploy público de container único (Hugging Face Spaces, Render...):
# o agendador roda em segundo plano desde a subida do container (não depende de ninguém
# abrir o painel) e, como o disco é efêmero, uma imagem de exemplo entra por execução
# (modo demonstração). O docker-compose sobrescreve tudo isso.
ENV RUN_SCHEDULER_IN_UI=0 \
    DEMO_SEED_PER_RUN=1 \
    PIPELINE_INTERVAL_SECONDS=60 \
    MALLOC_ARENA_MAX=2
# MALLOC_ARENA_MAX=2: o glibc cria uma arena de memória por thread e não as devolve; em
# hospedagem com 512 MB isso derruba o container.
# `nice`: o agendador roda com prioridade baixa de CPU, para o painel e o health check
# continuarem respondendo enquanto as imagens são processadas (instâncias com CPU fracionada).
# PORT é definido por plataformas como o Render; sem ele, usa 8501.
CMD ["sh", "-c", "nice -n 15 python -m scheduler.run_pipeline --loop & exec streamlit run src/ui/app.py --server.port=${PORT:-8501} --server.address=0.0.0.0 --server.headless=true --browser.gatherUsageStats=false"]
