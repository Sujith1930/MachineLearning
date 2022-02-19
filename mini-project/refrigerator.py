# -*- coding: utf-8 -*-
"""
Created on Wed Feb 16 18:50:29 2022

@author: snair
"""

import os
import mysql.connector


mydb = mysql.connector.connect(host="localhost", user="root", passwd="root", database="refrigerator")

#print(mydb)
mycursor = mydb.cursor()

mycursor.execute("select food_name, use_by_date, is_deleted from foods where use_by_date = '2021-05-18'")

result = mycursor.fetchall()



for res in result:
    print(res)
    sql = "select recipe_name from recipes where recipe_name like '%{}%'"
    mycursor.execute(sql.format(res[0]))
    result1 = mycursor.fetchall()
    print(result1)

