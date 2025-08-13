import os
from src.datascience import logger
import pandas as pd

from src.datascience.entity.config_entity import DataValidationConfig


class DataValiadtion:
    def __init__(self, config: DataValidationConfig):
        self.config = config

    def validate_all_columns(self)-> bool:
        try:
            validation_status = None

            data=pd.read_csv(self.config.data_tobe_validated, encoding='latin1')
            # Remove all columns that are completely empty (all values are NaN)
            data=data.drop(columns=['Unnamed: 2','Unnamed: 3','Unnamed: 2','Unnamed: 4'])
            data.rename(columns={'v1':'target','v2':'text'},inplace=True)
            all_cols = list(data.columns)

            all_schema = self.config.all_schema.keys()
            logger.info(f"all columns: {all_cols}")
            logger.info(f"all schema: {all_schema}")
            
            for col in all_cols:
                if col not in all_schema:
                    validation_status = False
                    with open(self.config.STATUS_FILE, 'w') as f:
                        f.write(f"Validation status: {validation_status}")
                else:
                    validation_status = True
                    with open(self.config.STATUS_FILE, 'w') as f:
                        f.write(f"Validation status: {validation_status}")
                    
            data.to_csv(self.config.validated_data, index=False)
            logger.info(f"Data validation completed successfully. Validated data saved at: {self.config.validated_data}")
            return validation_status
        
        except Exception as e:
            raise e

    

