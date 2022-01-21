FROM python:3

WORKDIR C:\Users\Felhasználó\Desktop\summer-projects\summer-projects\python_ML\linear_regression

COPY requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

CMD [ "python", "linear_regression.py" ]