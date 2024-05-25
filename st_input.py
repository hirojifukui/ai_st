import streamlit as st
# st.header("Streamlit Sample")
# st.title("Input area")

string = st.text_input("Enter a message")
st.write(string)

# num1 = st.number_input("Enter an integer", min_value = 0, max_value=100)
# st.write(num1)

# num2 = st.slider("Today's feeling", min_value = 0, max_value=100, step = 10)
# st.write(num2)

# st.button("Reset", type="primary")
# if st.button("Say hello"):
#     st.write("Hello Friend!")
# else:
#     st.write("Goodbye")

# is_agree = st.checkbox("Can you commit doing homework?")
# if is_agree:
#     st.write("Great!")
# else:
#     st.write("You better do it")

# options = st.multiselect(
#     "What are your favorite colors (Multiple selection)?",
#     ["Green", "Yellow", "Red", "Blue","Purple"],
#     ["Yellow", "Red"])
# st.write("You selected:", options)

# on = st.toggle("Activate feature", value=True)
# if on:
#     st.write("Feature activated!")
# else:
#     "Feature deactivated"

# import datetime
# today = datetime.datetime.now()
# next_year = today.year + 1
# jan_1 = datetime.date(next_year, 1, 1)
# dec_31 = datetime.date(next_year, 12, 31)

# period = st.date_input(
#     "Select your vacation for next year",
#     (jan_1, datetime.date(next_year, 1, 7)),
#     jan_1,
#     dec_31,
#     format="MM.DD.YYYY",
# )
# period

