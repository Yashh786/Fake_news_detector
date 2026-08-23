@echo off
echo ============================================================
echo   Fake News Detector - Environment Setup (Windows + GPU)
echo ============================================================

:: Create virtual environment
echo [1/5] Creating virtual environment...
python -m venv venv
call venv\Scripts\activate.bat

:: Upgrade pip
echo [2/5] Upgrading pip...
python -m pip install --upgrade pip

:: Install PyTorch with CUDA 12.1 (RTX 4060 compatible)
echo [3/5] Installing PyTorch with CUDA 12.1 support...
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121

:: Install remaining requirements
echo [4/5] Installing remaining packages...
pip install -r requirements.txt

:: Download NLTK data
echo [5/5] Downloading NLTK corpora...
python -c "import nltk; nltk.download('punkt'); nltk.download('stopwords'); nltk.download('wordnet'); nltk.download('vader_lexicon'); nltk.download('averaged_perceptron_tagger')"

:: Download spaCy English model
python -m spacy download en_core_web_sm

echo.
echo ============================================================
echo   Setup complete! Activate with: venv\Scripts\activate
echo ============================================================
pause
