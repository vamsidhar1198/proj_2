import streamlit as st
import pandas as pd
import numpy as np
widget_id = (id for id in range(100, 10000))
st.title('HELLO EVERYONE 😃')
st.header('I AM :red[VAMSIDHA REDDY YERUVA] 😎')
st.header('THIS IS MY FIRST STREAMLIT APP🤘')

code = '''print('HELLO WORLD!')'''
st.code(code,language='python')

st.header("LET'S HAVE SOME FUN HERE ")
'''can you solve this problem for me?'''
st.header('12 + 6 × 27 ÷ 3 + 2 – 16 ÷ 8 × 2')
ans = st.text_input('answer',key=1)
if ans =='64': 
   st.subheader('congrats 👏 you it right')
   st.balloons()
st.header('DO YOU LIKE PROGRAMMING ?')
yes = st.checkbox('YES',key=next(widget_id))
no = st.checkbox('NO',key=next(widget_id))
if yes:
   '''Are you familiar with python coding ?'''
   yes1 = st.checkbox('YESs',key=next(widget_id))
   no1 = st.checkbox('NOo',key=next(widget_id))
   if yes1:
      '''Let's try some questions 😜'''
      '''what could be output of this code'''
      '''x = [1,2,3,4,5,6]'''
      '''x[1:4]=[10]'''
      ''' x ?'''
      ans2 = st.text_input('answer')
      if ans2 == '[1,10,5,6]':
         '''your right'''
      elif ans2 == None:
         '''check again'''
   if no1:
      '''no problem lets meet again'''
if no:
   '''no problem lets meet again'''



