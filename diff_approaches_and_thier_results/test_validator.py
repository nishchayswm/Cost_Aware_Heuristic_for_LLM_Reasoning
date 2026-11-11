from validator import safe_eval, check_equation

print(safe_eval("12*4+3"))               
print(safe_eval("2**3"))                 
print(check_equation("12*4 = 48"))       
print(check_equation("12*4 = 50"))       
print(check_equation("so we get 48"))    