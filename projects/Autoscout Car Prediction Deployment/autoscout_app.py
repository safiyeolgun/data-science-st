import streamlit as st
import pandas as pd
import pickle
import time

# sidebar creating
sidebar_title = """<div style="background-color:tomato;">
<h3 style="color:white;text-align:center;">Car Features</h3>
</div><br>"""
st.sidebar.markdown(sidebar_title, unsafe_allow_html=True)

sidebar_message = """<p style="color:tomato; font-weight:bold; text-shadow: 2px 2px 5px #000000;">
Please make the selections below to make a car price estimate.</p>
"""
st.sidebar.markdown(sidebar_message, unsafe_allow_html=True )

# reading dataset for min, max and mode
car = pd.read_csv("final_scout_not_dummy.csv")

# creating selectbox for make_model
car_make_model = tuple(sorted(car["make_model"].unique()))
#car_model = st.sidebar.selectbox("Select make and model of your car:", ["Make & model"] + list(car_make_model))
car_model = st.sidebar.selectbox("Select make and model of your car:", [None] + list(car_make_model))
st.sidebar.markdown("<br>", unsafe_allow_html=True)

# creating radiobuton for gear
car_gearing_type = tuple(sorted(car["Gearing_Type"].unique()))
gearing_type = st.sidebar.radio("Select gear type of your car:", [None] + list(car_gearing_type))
st.sidebar.markdown("<br>", unsafe_allow_html=True)

# creating selectbox for age
car_age = tuple(sorted(car["age"].astype(int).unique()))
age = st.sidebar.selectbox("Select age of your car:", [None] + list(car_age))
st.sidebar.markdown("<br>", unsafe_allow_html=True)

# creating slider for km
car_km = int(car["km"].min()), int(car["km"].max()), int(car["km"].value_counts().idxmax())
km = st.sidebar.slider("Select km of your car:", 
                       min_value = car_km[0], 
                       max_value = car_km[1], 
                       value = car_km[2], 
                       step=1, format="%.0f")
st.sidebar.write(f"You selected {km:,.0f}km")
st.sidebar.markdown("<br>", unsafe_allow_html=True)

# creating number input for engine power
#car_kw = int(car["hp_kW"].min()), int(car["hp_kW"].max()), int(car["hp_kW"].value_counts().idxmax())
#engine_power = st.sidebar.number_input(
#    "Select engine power of your car (kW): ",
#    min_value=car_kw[0],
#    max_value=car_kw[1],
#    value=car_kw[2]    
#)
#st.sidebar.markdown("<br>", unsafe_allow_html=True)


# creating number input for weight
#car_weight = int(car["Weight_kg"].min()), int(car["Weight_kg"].max()), int(car["Weight_kg"].value_counts().idxmax())
#weight = st.sidebar.number_input(
#    "Select weight of your car (kg): ",
#    min_value=car_weight[0],
#    max_value=car_weight[1],
#    value=car_weight[2]    
#)
#st.sidebar.markdown("<br>", unsafe_allow_html=True)


# creating radiobuton for type
car_type = tuple(sorted(car["Type"].unique()))
type = st.sidebar.radio("Select type of your car:", [None] + list(car_type))
st.sidebar.markdown("<br>", unsafe_allow_html=True)

# creating radiobuton for upholstery
car_upholstery = tuple(sorted(car["Upholstery_type"].unique()))
upholstery = st.sidebar.radio("Select upholstery type of your car:", [None] + list(car_upholstery))
st.sidebar.markdown("<br>", unsafe_allow_html=True)

# selecting predict model
regression_names = ["Linear Regression", "Lasso", "Ridge", "ElasticNet", "Decision Tree", "Random Forest","Support Vector"]

models = st.sidebar.multiselect("Select regression model(s) to prediction:", regression_names, default=regression_names[0])
st.sidebar.write(f"You selected {len(models)} model(s)")


# main page creating
page_title = """
<div style="background-color:green;padding:10px">
<h1 style="color:white;text-align:center;"> Car Price Prediction </h1>
</div>"""
st.markdown(page_title, unsafe_allow_html = True)
st.markdown("<br>", unsafe_allow_html=True)
test_car = {
    "make_model": car_model,
    "Gearing_Type":gearing_type,
    "age": age,
    "km": km,
    #"hp_kW": engine_power,
    #"Weight_kg": weight, 
    "Type": type, 
    "Upholstery_type":upholstery 
}

# converting sidebar to dataframe
df = pd.DataFrame.from_dict([test_car])
#col = ["MakeModel", "Gear", "Age", "km", "Power", "Weight", "Type", "Upholstery"]
col = ["MakeModel", "Gear", "Age", "km", "Type", "Upholstery"]
df1 = df.copy()
df1.columns = col
#df1.set_index("MakeModel", inplace=True)
st.table(df1.head())

#st.dataframe(df1.head(), hide_index=True)
#st.write(df1.head(), show_index=False)



# loading  models
linear_regression = pickle.load(open("linear_regression", "rb"))
lasso_regression = pickle.load(open("lasso_regression", "rb"))
ridge_regression = pickle.load(open("ridge_regression", "rb"))
elastic_regression = pickle.load(open("elastic_regression", "rb")) 
decision_tree_regression = pickle.load(open("decision_tree_regression", "rb"))
random_forest_regression = pickle.load(open("random_forest_regression", "rb"))
support_vector_regression = pickle.load(open("support_vector_regression", "rb"))

# creating predict button
predict = st.button("Predict", help="Click to predict")


# predicting
fields = {
    "make model": car_model,
    "gearing type": gearing_type,
    "age": age,
    "type": type,
    "upholstery": upholstery,
    "models": models
}


regression_models = {
    "Linear Regression": linear_regression,
    "Lasso": lasso_regression,
    "Ridge": ridge_regression,
    "ElasticNet": elastic_regression,
    "Decision Tree": decision_tree_regression,
    "Random Forest": random_forest_regression,
    "Support Vector": support_vector_regression
    
}

if predict:
    missing_fields = [key for key, value in fields.items() if value is None]
    missing_models = [model for model in models if model is None]
    if missing_fields or missing_models:
        st.warning("Please select the following field(s)")
        if missing_fields:
            for field in missing_fields:
                st.markdown(f"**{field.replace('_', ' ').title()} is not selected!**", unsafe_allow_html=True)
        if missing_models:
            st.markdown(f"**{model} is not selected!**", unsafe_allow_html=True)
    
        
    else:
        #result = linear_regression.predict(df)        
        with st.spinner('Wait for it...'):
           time.sleep(5)
          #  st.success(f"${result[0]:.2f}")
        results = []
        dict = {}
        
        st.write("Predictions:")
        for model_name in models:
            model = regression_models[model_name]
            result = model.predict(df)
            results.append((model_name, result[0]))
            # st.success(f"${result[0]:.2f}")
            
        for model_name, prediction in results:
            #st.write(f"{model_name}: ${prediction:.2f}")
            #st.success(f"{model_name}: ${prediction:.2f}")
            dict[model_name] = prediction

        st.table(pd.DataFrame(dict, index=[0]))

# Tahminleri alın
#predictions = []
#for model_name in models:
#    model = regression_models[model_name]
#    result = model.predict(df)
#    predictions.append(result[0])

# Tahminleri DataFrame'e ekleme
#df['Predictions'] = predictions
#df['Predictions'] = pd.Series(predictions, index=[0,1]) #df.index


# Streamlit arayüzünde tahminleri gösterme
#st.write(df[['make_model', 'Gearing_Type', 'age', 'km', 'Type', 'Upholstery_type', 'Predictions']])



