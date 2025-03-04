# What is it

DocSage is personalized chatbot to read data from inserted files and then provide answers to your queries.

I have test on Python 3.10 only.

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


### Version 1
A terminal version of app has been made.

To run :-

```
python termapp.py ./location/to/pdf -q "Ask question related to pdf provided"
```


### Version 2

A GUI version of app has been made using streamlit

To run :-

If running on native platform
```bash
streamlit run app.py
```

If running on WSL2 via Windows use below command
```bash
streamlit run app.py --server.headless true
```
or below 

```bash
streamlit run app.py --server.address=0.0.0.0 --server.port=8501 --server.headless true\
``` 
