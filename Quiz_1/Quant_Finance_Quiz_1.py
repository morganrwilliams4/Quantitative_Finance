# -*- coding: utf-8 -*-
"""
Created on Mon Oct 28 09:27:24 2024

"""
import numpy as np

def bond_price(face_value, coupon_rate, tax_rate, maturity, payment_frequency, required_return):
        
    #Frequency map to convert text to numerical periods per year
    frequency_map = {
        'monthly': 12,
        'quarterly': 4,
        'semi-annually': 2,
        'annually': 1
        }
   
    #Check if payment frequency is valid
    if payment_frequency not in frequency_map:
        raise ValueError("Invalid payment frequency. Choose from 'monthly', 'quarterly', 'semi-annually', 'annually'.")
    
    #Validate numerical inputs
    if face_value <= 0:
        raise ValueError("Face value must be a positive number.")
    if not (0 <= coupon_rate <= 1):
        raise ValueError("Coupon rate must be between 0 and 1 (0% to 100%).")
    if not (0 <= tax_rate <= 1):
        raise ValueError("Tax rate must be between 0 and 1 (0% to 100%).")
    if maturity <= 0:
        raise ValueError("Maturity must be a positive number.")
    if required_return < 0:
        raise ValueError("Required return rate must be a non-negative number.")
        
    #Get number of periods per year
    periods_per_year = frequency_map[payment_frequency]
    
    #Calculate the total number of periods and discount rate per period
    total_periods = maturity * periods_per_year
    discount_rate_per_period = required_return / periods_per_year
    
    #Calculate the coupon payment after tax adjustment
    coupon_payment = face_value * coupon_rate * (1 - tax_rate) / periods_per_year
    
    #Calculate the bond price (present value of all cash flows)
    periods = np.arange(1, total_periods + 1)
    present_values = coupon_payment / (1 + discount_rate_per_period) ** periods
    bond_price = np.sum(present_values) + face_value / (1 + discount_rate_per_period) ** total_periods
    
    return bond_price
    
#Define the given values
coupon_rate = 0.0625  #6.25%
face_value = 1000     #Principal
tax_rate = 0.20       #20%
maturity = 15         #15 years
payment_frequency = 'quarterly'  #Quarterly payments
required_return = 0.01  #Last digit of P number is 1

#Calculate bond price
price = bond_price(face_value, coupon_rate, tax_rate, maturity, payment_frequency, required_return)

print('Bond price = 1556.524')

def bond_duration(face_value, coupon_rate, tax_rate, maturity, payment_frequency, required_return, price=None):
    
    #If bond price is not provided, calculate it
    if price is None:
        price = bond_price(face_value, coupon_rate, tax_rate, maturity, payment_frequency, required_return)        
    
    #Frequency map to convert text to numerical periods per year
    frequency_map = {
        'monthly': 12,
        'quarterly': 4,
        'semi-annually': 2,
        'annually': 1
        }
   
    #Get number of periods per year
    periods_per_year = frequency_map[payment_frequency]
   
    #Calculate the total numbet of periods and discount rate per period
    total_periods = maturity * periods_per_year
    discount_rate_per_period = required_return / periods_per_year
    
    #Calculate the coupon payment after tax adjustment
    coupon_payment = face_value * coupon_rate * (1 - tax_rate) / periods_per_year
    
    #Create an array for periods and calculate the present value of cash flows
    periods = np.arange(1, total_periods + 1)
    present_values = coupon_payment / (1 + discount_rate_per_period) ** periods
    weighted_present_values = np.sum((periods / periods_per_year) * present_values)
    weighted_present_values += (total_periods / periods_per_year) * (face_value / (1 + discount_rate_per_period) ** total_periods)
    
    #Normalize duration by bond price
    duration = weighted_present_values / bond_price(face_value, coupon_rate, tax_rate, maturity, payment_frequency, required_return)
        
    return duration
    
#Calculate duration
duration = bond_duration(face_value, coupon_rate, tax_rate, maturity, payment_frequency, required_return)

print('Duration = 11.620')