import json
import csv
import glob
import pandas as pd

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


"""    Read Patients Data    """

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
fieldnames = ['Patient ID', 'Patient Name', 'Gender', 'Birth Date', 'Address']
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

        # Write data to CSV
        writer.writerow({
            'Patient ID': patient_id,
            'Patient Name': patient_name,
            'Gender': gender,
            'Birth Date': birth_date,
            'Address': address
        })

        parsedPatientsData.append({
            'Patient ID': patient_id,
            'Patient Name': patient_name,
            'Gender': gender,
            'Birth Date': birth_date,
            'Address': address
        })
        
patients_df = pd.DataFrame(parsedPatientsData) #convert to dataframe

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


# Merge encounters with procedures
encounters_procedures_df = pd.merge(encounters_df, procedures_df, on=['Encounter ID', 'Patient ID'], how='left')

# Merge encounters with patients
encounters_patients_df = pd.merge(encounters_procedures_df, patients_df, on='Patient ID', how='left')

# Merge encounters with conditions
merged_data = pd.merge(encounters_patients_df, conditions_df, on='Patient ID', how='left')

merged_data.to_csv('mergedEncountersData.csv', index=False)


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