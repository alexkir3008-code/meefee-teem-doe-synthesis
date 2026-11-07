# Нейросетевой синтез фазового ДОЭ (MEEFEE TEEM, Nuclear IT Hack 2026)
# Сборка:  docker build -t meefee-doe .
# Запуск:  docker run --rm -v <папка с комплектами>:/data/input -v <папка для результатов>:/data/output meefee-doe
FROM python:3.12-slim

ENV PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

WORKDIR /app

# PyTorch для процессора (работает на любом компьютере, видеокарта не нужна),
# затем остальные зависимости из requirements.txt
COPY requirements.txt .
RUN pip install torch --index-url https://download.pytorch.org/whl/cpu \
 && pip install -r requirements.txt

COPY doe ./doe
COPY models ./models
COPY examples ./examples
COPY infer.py selftest.py ./

ENTRYPOINT ["python", "infer.py"]
CMD ["--input", "/data/input", "--output", "/data/output"]
