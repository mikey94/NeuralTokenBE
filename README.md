# NeuralTokenBE
Created for Kingston University's master's research project. This repo serves as a backend for a CryptoCurrency price prediction ML research project. 

<h2>Instructions to run the backend</h2>

1) Clone the repository.

```bash
git clone {repository name}
```
  
2) Install the modules inside the requirements.txt

```bash
pip install -r requirements.txt
```

3) Verify the installation.

```bash
pip list
```

4) Run the application

```bash
uvicorn main:app --reload --host 0.0.0.0 --port 8000 
```
