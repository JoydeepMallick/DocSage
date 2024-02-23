# What is it

DocSage is personalized chatbot to read data from inserted files and then provide answers to your queries.

To run locally,

1. install conda/miniconda

```
conda create --name myenv python=3.10
```

2. Install all dependencies

```
pip install -r requirements.txt
```

3. put in your API key in `.env` file. Get your api key from [here](https://makersuite.google.com/app/apikey)

Right now a terminal version of app has been made.

To run :-

```
python termapp.py ./location/to/pdf -q "Ask question related to pdf provided"
```

It fails to parse big pdf files sadly😦.
