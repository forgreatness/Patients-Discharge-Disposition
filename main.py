import json
import csv
import glob
import pandas as pd
import numpy as np

from sklearn.preprocessing import LabelEncoder, MultiLabelBinarizer
from datetime import datetime

label_encoder = LabelEncoder()

# List all encounter files
encounter_files = glob.glob("./data/encounters/*.ndjson")

# List to store encounter data from all files
encounters_data = []

# Iterate over each encounter file
for file in encounter_files:
    # Open the NDJSON file and read lines
    with open(file, "r") as f:
        lines = f.readlines()

    # Parse each line as JSON and append to encounters_data list
    for line in lines:
        encounter = json.loads(line)
        encounters_data.append(encounter)

# Define CSV file and fieldnames
csv_file = "encounters.csv"
fieldnames = ['Encounter ID', 'Patient ID', 'Start Date', 'End Date', 'Encounter Type', 'Service Type', 'Reason']
parsedEncountersData = []

# Write data to CSV file
with open(csv_file, mode='w', newline='') as f:
    writer = csv.DictWriter(f, fieldnames=fieldnames)
    writer.writeheader()

    # Loop through each encounter data
    for encounter in encounters_data:
        # Extract relevant data
        encounter_id = encounter.get('id', '')
        patient_id = encounter.get('subject', {}).get('reference', '').split('/')[-1]
        start_date = encounter.get('period', {}).get('start', '')
        end_date = encounter.get('period', {}).get('end', '')
        encounter_type = encounter.get('type', [{}])[0].get('text', '')
        service_type = encounter.get('serviceType', {}).get('coding', [{}])[0].get('display', '')
        reason = encounter.get('reasonCode', [{}])[0].get('text', '')

        # Write data to CSV
        writer.writerow({
            'Encounter ID': encounter_id,
            'Patient ID': patient_id,
            'Start Date': start_date,
            'End Date': end_date,
            'Encounter Type': encounter_type,
            'Service Type': service_type,
            'Reason': reason
        })
        
        # Append data as a dictionary to the list
        parsedEncountersData.append({
            'Encounter ID': encounter_id,
            'Patient ID': patient_id,
            'Start Date': start_date,
            'End Date': end_date,
            'Encounter Type': encounter_type,
            'Service Type': service_type,
            'Reason': reason
        })
        
encounters_df = pd.DataFrame(parsedEncountersData) #convert to dataframe
encounters_df = encounters_df.drop_duplicates(subset=['Encounter ID']) #remove duplicates


"""    Read Procedures Data    """

# List all patient files
procedure_files = glob.glob("./data/procedures/1.Procedure*.ndjson")

# List to store patient data from all files
procedures_data = []

# Iterate over each patient file
for file in procedure_files:
    # Open the NDJSON file and read lines
    with open(file, "r") as f:
        lines = f.readlines()

    # Parse each line as JSON and append to patients_data list
    for line in lines:
        procedure = json.loads(line)
        procedures_data.append(procedure)

# Define CSV file and fieldnames
csv_file = "procedures.csv"
fieldnames = ['Procedure ID', 'Patient ID', 'Patient Name', 'Encounter ID', 'Performed Start Date', 'Performed End Date', 'Procedure Type']
parsedProceduresData = []

# Write data to CSV file
with open(csv_file, mode='w', newline='') as file:
    writer = csv.DictWriter(file, fieldnames=fieldnames)
    writer.writeheader()

    # Loop through each line of NDJSON
    for procedure in procedures_data:
        # Extract relevant data
        procedure_id = procedure.get('id', '')
        patient_id = procedure.get('subject', {}).get('reference', '').split('/')[-1]
        patient_name = procedure.get('subject', {}).get('display', '')
        encounter_id = procedure.get('encounter', {}).get('reference', '').split('/')[-1]
        performed_start_date = procedure.get('performedPeriod', {}).get('start', '')
        performed_end_date = procedure.get('performedPeriod', {}).get('end', '')
        procedure_type = procedure.get('code', {}).get('text', '')

        # Write data to CSV
        writer.writerow({
            'Procedure ID': procedure_id,
            'Patient ID': patient_id,
            'Patient Name': patient_name,
            'Encounter ID': encounter_id,
            'Performed Start Date': performed_start_date,
            'Performed End Date': performed_end_date,
            'Procedure Type': procedure_type
        })
        
        # Append data as a dictionary to the list
        parsedProceduresData.append({
            'Procedure ID': procedure_id,
            'Patient ID': patient_id,
            'Patient Name': patient_name,
            'Encounter ID': encounter_id,
            'Performed Start Date': performed_start_date,
            'Performed End Date': performed_end_date,
            'Procedure Type': procedure_type
        })

procedures_df = pd.DataFrame(parsedProceduresData) #convert to dataframe
procedures_df = procedures_df.drop_duplicates(subset=['Procedure ID']) #remove duplicates


"""    Read Patients Data    """
# Function to parse ethnicity from the patient data
def parse_ethnicity(patient):
    """
    Parse ethnicity information from patient data.
    """
    ethnicity = None
    extensions = patient.get('extension', [])
    for ext in extensions:
        if ext.get('url') == 'http://hl7.org/fhir/us/core/StructureDefinition/us-core-ethnicity':
            ethnicity_ext = ext.get('extension', [])
            for eth_ext in ethnicity_ext:
                if eth_ext.get('url') == 'Text':
                    ethnicity = eth_ext.get('valueString', None)
    return ethnicity

# List all patient files
patient_files = glob.glob("./data/patients/1.Patient*.ndjson")

# List to store patient data from all files
patients_data = []

# Iterate over each patient file
for file in patient_files:
    # Open the NDJSON file and read lines
    with open(file, "r") as f:
        lines = f.readlines()

    # Parse each line as JSON and append to patients_data list
    for line in lines:
        patient = json.loads(line)
        patients_data.append(patient)

# Define CSV file and fieldnames
csv_file = "patients.csv"
fieldnames = ['Patient ID', 'Patient Name', 'Gender', 'Birth Date', 'Address', 'Ethnicity']
parsedPatientsData = []

# Write data to CSV file
with open(csv_file, mode='w', newline='') as f:
    writer = csv.DictWriter(f, fieldnames=fieldnames)
    writer.writeheader()

    # Loop through each patient data
    for patient in patients_data:
        # Extract relevant data
        patient_id = patient.get('id', '')
        patient_name = patient.get('name', [{}])[0].get('given', [''])[0] + " " + patient.get('name', [{}])[0].get('family', '')
        gender = patient.get('gender', '')
        birth_date = patient.get('birthDate', '')
        address = ', '.join(patient.get('address', [{}])[0].get('line', []))
        ethnicity = parse_ethnicity(patient)  # Parse ethnicity


        # Write data to CSV
        writer.writerow({
            'Patient ID': patient_id,
            'Patient Name': patient_name,
            'Gender': gender,
            'Birth Date': birth_date,
            'Address': address,
            'Ethnicity': ethnicity
        })

        parsedPatientsData.append({
            'Patient ID': patient_id,
            'Patient Name': patient_name,
            'Gender': gender,
            'Birth Date': birth_date,
            'Address': address,
            'Ethnicity': ethnicity
        })
        
patients_df = pd.DataFrame(parsedPatientsData) #convert to dataframe
patients_df = patients_df.drop_duplicates(subset=['Patient ID']) #remove duplicates

"""       Read Conditions data        """

# List all condition files
condition_files = glob.glob("./data/conditions/*.ndjson")

# List to store condition data from all files
conditions_data = []

# Iterate over each condition file
for file in condition_files:
    # Open the NDJSON file and read lines
    with open(file, "r") as f:
        lines = f.readlines()

    # Parse each line as JSON and append to conditions_data list
    for line in lines:
        condition = json.loads(line)
        conditions_data.append(condition)

# Define CSV file and fieldnames
csv_file = "conditions.csv"
fieldnames = ['Condition ID', 'Patient ID', 'Code', 'Onset Date', 'Note']
parsedConditionsData = []

# Write data to CSV file
with open(csv_file, mode='w', newline='') as f:
    writer = csv.DictWriter(f, fieldnames=fieldnames)
    writer.writeheader()

    # Loop through each condition data
    for condition in conditions_data:
        # Extract relevant data
        condition_id = condition.get('id', '')
        patient_id = condition.get('subject', {}).get('reference', '').split('/')[-1]
        code = condition.get('code', {}).get('text', '')
        onset_date = condition.get('onsetDateTime', '')
        note = condition.get('note', [{}])[0].get('text', '')

        # Write data to CSV
        writer.writerow({
            'Condition ID': condition_id,
            'Patient ID': patient_id,
            'Code': code,
            'Onset Date': onset_date,
            'Note': note
        })
        
        parsedConditionsData.append({
            'Condition ID': condition_id,
            'Patient ID': patient_id,
            'Code': code,
            'Onset Date': onset_date,
            'Note': note
        })

conditions_df = pd.DataFrame(parsedConditionsData) #convert to dataframe
conditions_df = conditions_df.drop_duplicates(subset=['Condition ID']) #remove duplicates

# Merge encounters with patients
merged_data = pd.merge(encounters_df, patients_df, on='Patient ID', how='left')

# Initialize an empty list to store the condition codes for each row
condition_codes_list = []

# Iterate through each row in the merged_data dataframe
for index, row in merged_data.iterrows():
    # Filter conditions dataframe based on the current patient ID
    patient_conditions = conditions_df[conditions_df['Patient ID'] == row['Patient ID']]
    
    # Extract relevant condition codes and store them in a list
    condition_codes = patient_conditions['Code'].tolist()
    
    # Append the list of condition codes to the condition_codes_list
    condition_codes_list.append(condition_codes)

# Add a new column named 'Condition Codes' to the merged_data dataframe
merged_data['Conditions'] = condition_codes_list

# Initialize an empty list to store the procedure types for each row
procedure_types_list = []

# Iterate through each row in the merged_data dataframe
for index, row in merged_data.iterrows():
    # Filter procedures dataframe based on the current patient ID
    patient_procedures = procedures_df[procedures_df['Patient ID'] == row['Patient ID']]
    
    # Extract relevant procedure types and store them in a list
    procedure_types = patient_procedures['Procedure Type'].tolist()
    
    # Append the list of procedure types to the procedure_types_list
    procedure_types_list.append(procedure_types)

# Add a new column named 'Procedure Types' to the merged_data dataframe
merged_data['Procedures'] = procedure_types_list

"""    Data Preprocessing     """
# Step #1: Replace any data that is null | undefined | empty with nan
merged_data.replace('?', np.nan, inplace=True)
nanCountPerFeature = merged_data.isna().sum()
featuresHavingTooMuchNan = nanCountPerFeature[nanCountPerFeature > 50].index.values
merged_data.drop(featuresHavingTooMuchNan, axis=1, inplace=True)

# Step #2: Remove all rows with nan data
merged_data = merged_data.dropna() #remove all rows where there is a column with nan value

# Step #3: Convert encounter start date and encounter end date to duration of care
# Convert 'Start Date' and 'End Date' to datetime objects
# Convert 'Start Date' and 'End Date' to datetime objects
merged_data['Start Date'] = pd.to_datetime(merged_data['Start Date'])
merged_data['End Date'] = pd.to_datetime(merged_data['End Date'])

# Calculate duration of care in days
merged_data['Duration of Care'] = (merged_data['End Date'] - merged_data['Start Date'])
merged_data['Duration of Care'] = merged_data['Duration of Care'].apply(lambda x: x.days)

# #Step #4 Convert the Birth Date to age
for index, row in merged_data.iterrows():
    # Convert birth date string to datetime object
    birth_date = datetime.strptime(row['Birth Date'], "%Y-%m-%d")
    
    # Calculate age
    current_date = datetime.now()
    age = current_date.year - birth_date.year - ((current_date.month, current_date.day) < (birth_date.month, birth_date.day))
    
    # Assign the age to a new column in the DataFrame
    merged_data.at[index, 'Age'] = int(age)

# Convert the 'Age' column to integer type
merged_data['Age'] = merged_data['Age'].astype(int)

# Step 5: Label encode the gender
# Fit and transform the 'Gender' and 'Service Type' column
merged_data['Gender'] = label_encoder.fit_transform(merged_data['Gender'])
merged_data['Service Type'] = label_encoder.fit_transform(merged_data['Service Type'])
merged_data['Ethnicity'] = label_encoder.fit_transform(merged_data['Ethnicity'])
merged_data.reset_index(drop=True, inplace=True)

# Step 6: Encode the array of conditions and array of procedures
# unique_conditions = list(set(condition for sublist in merged_data['Conditions'] for condition in sublist))

# # Create a binary matrix for one-hot encoding
# one_hot_encoded = pd.DataFrame(0, columns=unique_conditions, index=merged_data.index)

# # Loop through each encounter and fill the one-hot encoded matrix
# for idx, encounter_conditions in enumerate(merged_data['Conditions']):
#     if encounter_conditions:  # Check if there are conditions for this encounter
#         for condition in encounter_conditions:
#             one_hot_encoded.at[idx, condition] = 1

# # Convert one_hot_encoded to a list of lists
# encoded_conditions_list = one_hot_encoded.values.tolist()
# # Add the list of encoded conditions to merged_data
# merged_data['Encoded_Conditions'] = encoded_conditions_list
# # Drop the original 'Conditions' column
# merged_data.drop('Conditions', axis=1, inplace=True)

# # ####### Encode the array of procedures #######
# unique_procedures = list(set(procedure for sublist in merged_data['Procedures'] for procedure in sublist))

# # Create a binary matrix for one-hot encoding
# one_hot_encoded_procedures = pd.DataFrame(0, columns=unique_procedures, index=merged_data.index)

# # Loop through each encounter and fill the one-hot encoded matrix
# for idx, encounter_procedures in enumerate(merged_data['Procedures']):
#     if encounter_procedures:  # Check if there are conditions for this encounter
#         for procedure in encounter_procedures:
#             one_hot_encoded_procedures.at[idx, procedure] = 1


# # Convert one_hot_encoded to a list of lists
# encoded_procedures_list = one_hot_encoded_procedures.values.tolist()

# # Add the list of encoded conditions to merged_data
# merged_data['Encoded_Procedures'] = encoded_conditions_list

# # Drop the original 'Conditions' column
# merged_data.drop('Procedures', axis=1, inplace=True)

# merged_data[['Encoded_Conditions', 'Encoded_Procedures']] = merged_data[['Encoded_Conditions', 'Encoded_Procedures']].applymap(lambda x: np.array(x).astype(int))

# Step 7: Feature selection
selectedFeatures = np.array(['Ethnicity','Duration of Care','Age', 'Procedures', 'Conditions', 'Service Type', 'Gender'])
preprocessedEncounters = merged_data[selectedFeatures]

# Extract unique procedures and conditions
all_procedures = set()
all_conditions = set()
for procedures, conditions in zip(preprocessedEncounters['Procedures'], preprocessedEncounters['Conditions']):
    all_procedures.update(procedures)
    all_conditions.update(conditions)

# Create binary matrices for procedures and conditions
mlb = MultiLabelBinarizer(classes=sorted(all_procedures.union(all_conditions)))
procedure_matrix = mlb.fit_transform(preprocessedEncounters['Procedures'])
condition_matrix = mlb.transform(preprocessedEncounters['Conditions'])

# Combine matrices with the original DataFrame
# Create DataFrame for one-hot encoded procedures and conditions
procedure_df = pd.DataFrame(procedure_matrix, columns=mlb.classes_)
condition_df = pd.DataFrame(condition_matrix, columns=mlb.classes_)

# Concatenate one-hot encoded data with original DataFrame
encoded_data = pd.concat([preprocessedEncounters.drop(columns=['Procedures', 'Conditions']), procedure_df, condition_df], axis=1)

# Step 8: Write it to final data then we can smote later and feature scale later
# preprocessedEncounters.to_csv('finalEncounterData.csv')
encoded_data.to_csv('finalEncounterData.csv')





























# # List all observation files
# observation_files = glob.glob("./data/observations/*.ndjson")

# # List to store observation data from all files
# observations_data = []

# # Iterate over each observation file
# for file in observation_files:
#     # Open the NDJSON file and read lines
#     with open(file, "r") as f:
#         lines = f.readlines()

#     # Parse each line as JSON and append to observations_data list
#     for line in lines:
#         observation = json.loads(line)
#         observations_data.append(observation)

# # Define CSV file and fieldnames
# csv_file = "observations.csv"
# fieldnames = ['Observation ID', 'Patient ID', 'Effective Date Time', 'Value Quantity', 'Unit']

# # Write data to CSV file
# with open(csv_file, mode='w', newline='') as f:
#     writer = csv.DictWriter(f, fieldnames=fieldnames)
#     writer.writeheader()

#     # Loop through each observation data
#     for observation in observations_data:
#         # Extract relevant data
#         observation_id = observation.get('id', '')
#         patient_id = observation.get('subject', {}).get('reference', '').split('/')[-1]
#         effective_date_time = observation.get('effectiveDateTime', '')
#         value_quantity = observation.get('valueQuantity', {}).get('value', '')
#         unit = observation.get('valueQuantity', {}).get('unit', '')

#         # Write data to CSV
#         writer.writerow({
#             'Observation ID': observation_id,
#             'Patient ID': patient_id,
#             'Effective Date Time': effective_date_time,
#             'Value Quantity': value_quantity,
#             'Unit': unit
#         })