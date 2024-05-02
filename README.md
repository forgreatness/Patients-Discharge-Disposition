# Purpose
The purpose of this document is to describe the thought process behind the intelligent model that takes in information relating to patients encounter to predict future encounter disposition

## Dataset
Each section below describe the object data point that are related to each encounter.

**Encounter**
- Encounter ID
- Patient ID
- Patient Name 
- Encounter Type
- Service Type
- Encounter Start Date
- Encounter End Date
- Reason For Encounter

**Patient**
- Patient ID
- Patient Name
- Gender
- Birth Date
- Race (ethnicity)

**Conditions**
Condition is clearly by patient, so 1 patient can have multiple condition. What we should do is for each patient, find all their condition in conditions first
- Condition ID
- Patient ID
- Code
- Onset Date
- Note

**Procedure**
An encounter can sometime have multiple procedure, we can also establish a procedures column when merging by Encounter ID
- Procedure ID
- Patient ID
- Patient Name
- Encounter ID
- Performed Start Date
- Performed End Date
- Procedure Type


**Overall Metrics**
- Age
- Gender
- Ethnicity
- Time of Care
- Service Type
- List of Condition (encoded)
- list of Procedure (encoded)

> [!IMPORTANT]
> To get the overal features we will merge patient data into encounter data by "Encounter ID"
> Then for each encounter we will loop through the set of data from procedures and conditions to get all the conditions and procedures for that encounter
> After that we will write this dataframe into a file call finalData.csv

### Preprocessing
1. Load the data into a dataframe (have column label for them all)
2. Convert any null | undefined | empty data box into NAN 
3. Check to see if there are any column with too many null and undefined data and remove that column
4. Check to see if there are any row that has empty data and also remove that
5. Convert the Encounter Start Date and the Encounter End Date into Time of Care
6. Encode the gender, the race, service type, list of condition, list of procedure
7. Feature selected the encounters data into a variable ready for splitting
8. Split data
9. Feature Scale data
10. Data Sampling using SMOTE

### Model Integration
> [!NOTE]
> Try different model that are good for regression on the data (avoid linear since we know right away its not beneficial)


**Other can be consider**
- procedure performer