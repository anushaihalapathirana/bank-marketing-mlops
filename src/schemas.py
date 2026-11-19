from pydantic import BaseModel, Field, ConfigDict

class CustomerFearures(BaseModel):
    model_config = ConfigDict(extra="forbid")
    
    age: int = Field(..., ge=0, description="Age of the customer")
    job: str = Field(..., description="Job type of the customer")
    marital: str = Field(..., description="Marital status of the customer")
    education: str = Field(..., description="Education level of the customer")
    default: str = Field(..., description="Whether the customer has credit in default")
    housing: str = Field(..., description="Whether the customer has a housing loan")
    loan: str = Field(..., description="Whether the customer has a personal loan")
    contact: str = Field(..., description="Contact communication type")
    day_of_week: str = Field(..., description="Last contact day of the week")
    month: str = Field(..., description="Last contact month of the year")
    campaign: int = Field(..., ge=1, description="Number of contacts performed during this campaign for this client")
    pdays: int = Field(..., ge=-1, description="Number of days that passed by after the client was last contacted from a previous campaign (-1 means client was not previously contacted)")
    previous: int = Field(..., ge=0, description="Number of contacts performed before this campaign for this client")
    poutcome: str = Field(..., description="Outcome of the previous marketing campaign")
    
    emp_var_rate: float = Field(..., description="Employment variation rate - quarterly indicator", alias="emp.var.rate")
    cons_price_idx: float = Field(..., description="Consumer price index - monthly indicator", alias="cons.price.idx")
    cons_conf_idx: float = Field(..., description="Consumer confidence index - monthly indicator", alias="cons.conf.idx")
    euribor3m: float = Field(..., description="Euribor 3 month rate - daily indicator")
    nr_employed: float = Field(..., description="Number of employees - quarterly indicator", alias="nr.employed")
    
class PredictionResponse(BaseModel):
    model_config = ConfigDict(extra="forbid")
    
    predicted_class: int = Field(..., description="Predicted class (0 or 1)")
    probability: float = Field(..., ge=0.0, le=1.0, description="Predicted probability of having a subscription")   
    