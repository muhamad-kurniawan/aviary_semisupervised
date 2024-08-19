import re
from collections import defaultdict

reg_element = re.compile("[A-Z][a-z]*")
reg_st = re.compile("(?=.*?\d)\d*\.\d*|\d\d\d|\d\d|\d")

def list_duplicates(seq):
    tally = defaultdict(list)
    for i,item in enumerate(seq):
        tally[item].append(i)
    return [[key,locs] for key,locs in tally.items() if len(locs)>1]

def list_element_stoichiometri(list_molecule):  #create list of each element with its total stoichiometri
  final_data = []
  for mol in list_molecule:
    ls_element = []
    start_pos = []
    end_pos = []

    ls_element_st = []
    ls_element_st_final = []
    start_pos_st = []
    end_pos_st = []

    for m in reg_element.finditer(mol):
      ls_element.append(m.group())
      start_pos.append(m.start())
      end_pos.append(m.end())
    for x in reg_st.finditer(mol):
      ls_element_st.append(x.group())
      start_pos_st.append(x.start())
      end_pos_st.append(x.end())

    for n_el in range(len(ls_element)):
      stochio_value = 1
      if end_pos[n_el] in start_pos_st:
          for n in range(len(start_pos_st)):
              if start_pos_st[n] == end_pos[n_el]:
                  stochio_value = float(ls_element_st[n])
      else:
          pass
      ls_element_st_final.append(stochio_value)

    duplicates = list_duplicates(ls_element)

    if len(duplicates)>0: #check duplicate
      duplicate_elements = [x[0] for x in duplicates]
      elem_list_ = []
      stoic_list_ = []
      #create list element non duplicate
      for idx in range(len(ls_element)):
        if ls_element[idx] not in duplicate_elements:
          elem_list_.append(ls_element[idx])
          stoic_list_.append(ls_element_st_final[idx])
      #add processed duplicate to the list
      for duplicate_elem in duplicates:
        elem_list_.append(duplicate_elem[0])
        sum_stoich = sum([ls_element_st_final[z] for z in duplicate_elem[1]])
        stoic_list_.append(sum_stoich)
      #change main list
      ls_element = elem_list_
      ls_element_st_final = stoic_list_

    final_data.append([ls_element, ls_element_st_final])
  return final_data

def bracket_multiplication(elements):
    reg_bracket = re.compile("\)\d\d*\.\d*|\)\d\d\d|\)\d\d|\)\d")
    ls_multi_bracket = [] #store multiplication number to the elements inside bracket
    ls_start_pos = [] #close bracket position 
    for m in reg_bracket.finditer(elements):

        ls_multi_bracket.append(float(m.group()[1:]))
        ls_start_pos.append(m.start())

    ls_open_bracket = [] #open bracket position 
    for n in range(len(ls_multi_bracket)):
        another_bracket = False
        for num in range(ls_start_pos[n]):
            char_pos = ls_start_pos[n]-1-num
            if elements[char_pos] ==')':
                another_bracket = True
            if elements[char_pos] =='(' and another_bracket==False:
                ls_open_bracket.append(char_pos)
                break
            if elements[char_pos] =='(' and another_bracket==True:
                another_bracket = False
    
    return ls_multi_bracket, ls_open_bracket, ls_start_pos

def open_bracket_multi(molecules):
    reg_element = re.compile("[A-Z][a-z]*")
    reg_st = re.compile("(?=.*?\d)\d*\.\d*|\d\d\d|\d\d|\d")
    # reg_bracket = 
    
    ls_element = []
    start_pos = []
    end_pos = []

    ls_element_st = []
    start_pos_st = []
    end_pos_st = []

    total_stochiometri = []
    for m in reg_element.finditer(molecules):

        ls_element.append(m.group())
        start_pos.append(m.start())
        end_pos.append(m.end())
            
    for x in reg_st.finditer(molecules):
        ls_element_st.append(x.group())
        start_pos_st.append(x.start())
        end_pos_st.append(x.end())
        
    bracket_multi = bracket_multiplication(molecules)

    for n_el in range(len(ls_element)):
        multi_num = 1
        if end_pos[n_el] in start_pos_st:
            for n in range(len(start_pos_st)):
                if start_pos_st[n] == end_pos[n_el]:
                    multi_num = float(ls_element_st[n])        
        else:
            pass
        for n2 in range(len(bracket_multi[0])):
            if start_pos[n_el]> bracket_multi[1][n2] and start_pos[n_el] < bracket_multi[2][n2]:
                multi_num = multi_num*float(bracket_multi[0][n2])

        total_stochiometri.append(multi_num)
    
    final_formula = ''
    for elem in range(len(ls_element)):
        stochio = float(total_stochiometri[elem])
        final_formula += ls_element[elem]
        if stochio % 1 == 0:
            stochio = int(total_stochiometri[elem])
        if stochio != 1:
            final_formula += str(stochio)
    
    return final_formula
