FROM python:3.10-slim

WORKDIR /img_fol

COPY . .

COPY ["database_generator", "/img_fol/database"]

CMD ["python", "script.py"]

































FROM python:3.9-slim


WORKDIR /fol_2_2

ENV USER="Sadio Mane"

COPY file.py .

RUN apt-get update && apt-get install -y \
    htop \
    sudo \
    nano \
    && rm -rf /var/lib/apt/lists/*

ENTRYPOINT ["python", "file.py"]

CMD ["Not-Normal"]




























FROM python:3.9-slim


WORKDIR /fol_2_3

COPY requirements.txt .

RUN echo -e "//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////"

RUN pip install --no-cache-dir -r requirements.txt

COPY . .

CMD ["python", "file.py"]







