from dataclasses import dataclass, field

@dataclass
class ModelVariables:
    responding_variable: str = 'Sold_Amount'
    exploration_variables: list = field(default_factory=lambda: ["MakeCode", "FamilyCode", "Power", "SeatCapacity", "Sold_Date", "NewPrice", "Age_Comp_Months", "GearNum", "DoorNum", "EngineDescription", "Cylinders", "FuelTypeDescription", "FuelCapacity", "RonRating", "SeatCapacity", "BuildCountryOriginDescription", "WarrantyYears", "WarrantyKM", "FirstServiceKM", "FirstServiceMonths", "Age_Comp_Months","OverallGreenStarRating", "SaleCategory", "KM"]) # Variables to be included in exploration
    numeric_variables:list = field(default_factory=lambda: ["Power", "NewPrice", "GearNum", "DoorNum", "Cylinders", "SeatCapacity", "WarrantyYears", "WarrantyKM", "Age_Comp_Months", "KM"])
    categorical_variables: list = field(default_factory=lambda: ["MakeCode", "FamilyCode", "EngineDescription", "FuelTypeDescription", "RonRating", "BuildCountryOriginDescription", "SaleCategory"])
    date_variables: list = field(default_factory= lambda: ['Sold_Date'])
    engineered_variables: list = field(default_factory= lambda: ['Sold_Year', 'AvailableWarrantyYears', 'AvailableWarrantyKM'])
    excluded_variables: list = field(default_factory=lambda: ["Sold_Date", "WarrantyYears", "WarrantyKM"]) 