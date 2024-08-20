'''
###Importing the Libraries###
numpy as np (Used for numerical operations)
pandas as pd (Data manipulation and analysis)
ezodf (Reads .ods)
sklearn.model(Machine learning libraryfor data preprocessing & model evaluation)
tensorflow.keras (Library for training and building neural network models)
datetime (Used to handle time and date objects)
'''

import numpy as np
import pandas as pd
import ezodf
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from datetime import datetime

'''
###Function to Read ODS File###
read_ods (Reads an ODS file and converts it into a pandas DataFrame)
Error handling (If there's an error reading the file, it prints an error message and returns None)
'''

def read_ods(filename, sheet_no=0):
    try:
        doc = ezodf.opendoc(filename)
        sheet = doc.sheets[sheet_no]
        
        header_row = [cell.value for cell in sheet.row(0)]
        df = pd.DataFrame(columns=header_row)
        
        for i, row in enumerate(sheet.rows()):
            if i == 0:
                continue
            row_data = [cell.value for cell in row]
            row_data += [None] * (len(header_row) - len(row_data))
            df.loc[i-1] = row_data[:len(header_row)]
        
        return df
    except Exception as e:
        print(f"Error reading the file: {str(e)}")
        return None

'''
Load the data (Calls read_ods to load data from 'spending.ods'.)
Error check (If data loading fails, it prints an error message.)
'''
data = read_ods('spending.ods')

if data is None:
    print("Failed to read the data. Please check your file and try again.")
else:
    print("Data loaded successfully. Shape:", data.shape)
    print("Columns:", data.columns)

    # Restructure the data
    restructured_data = []
    for i in range(0, len(data.columns), 3):  # Assuming groups of 3 columns (Date, Amount, None)
        date_col = data.iloc[:, i]
        amount_col = data.iloc[:, i+1]
        for date, amount in zip(date_col, amount_col):
            if pd.notna(date) and pd.notna(amount):
                restructured_data.append([date, amount])

    df = pd.DataFrame(restructured_data, columns=['date', 'amount'])
    
    # Convert date to day of year
    def date_to_day_of_year(date_str):
        try:
            date = datetime.strptime(date_str, '%m/%d/%y')
            return date.timetuple().tm_yday
        except ValueError:
            print(f"Error parsing date: {date_str}")
            return None

    # Convert amount to float
    def amount_to_float(amount_str):
        try:
            return float(amount_str.replace('$', '').replace(',', ''))
        except ValueError:
            print(f"Error parsing amount: {amount_str}")
            return None

    df['day'] = df['date'].apply(date_to_day_of_year)
    df['amount'] = df['amount'].apply(amount_to_float)
    df = df.dropna()  # Remove any rows where conversion failed

    print("Restructured Data:")
    print(df.head())
    print("Shape:", df.shape)

    # Prepare the data
    X = df[['day', 'amount']].values
    y = df['amount'].values

    # Normalize the input features
    scaler = MinMaxScaler()
    X_scaled = scaler.fit_transform(X)

    # Split the data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

    # Create the neural network model
    model = Sequential([
        Dense(64, activation='relu', input_shape=(2,)),
        Dense(32, activation='relu'),
        Dense(16, activation='relu'),
        Dense(1)  # Output layer
    ])

    # Compile the model
    model.compile(optimizer='adam', loss='mean_squared_error')

    # Train the model
    history = model.fit(X_train, y_train, epochs=100, batch_size=32, validation_split=0.2, verbose=1)

    # Evaluate the model
    loss = model.evaluate(X_test, y_test)
    print(f"Test loss: {loss}")

    # Make predictions
    predictions = model.predict(X_test)
