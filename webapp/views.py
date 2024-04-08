import numpy as np
import os
import keras.utils as image
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.utils import load_img

from tensorflow.keras.models import load_model
from django.shortcuts import render,redirect
from django.shortcuts import get_object_or_404
from django.core.files.storage import FileSystemStorage
from django.http import HttpResponse
from django.forms import inlineformset_factory
from django.contrib.auth.forms import UserCreationForm
from django.contrib.auth import authenticate, login, logout
from django.contrib import messages
from django.contrib.auth.decorators import login_required
from .models import *
from .forms import CreateUserForm
import pickle

# Create your views here.
def registerPage(request):
    if request.user.is_authenticated:
        return redirect('home')
    else:
        form = CreateUserForm()
        if request.method == 'POST':
            form = CreateUserForm(request.POST)
            if form.is_valid():
                form.save()
                user = form.cleaned_data.get('username')
                messages.success(request, 'Account was created for ' + user)
                return redirect('login')

        context = {'form': form}
        return render(request, 'accounts/register.html', context)

def loginPage(request):
    if request.user.is_authenticated:
        return redirect('home')
    else:
        if request.method == 'POST':
            username = request.POST.get('username')
            password = request.POST.get('password')

            user = authenticate(request, username=username, password=password)
            if user is not None:
                login(request, user)
                return redirect('home')
            else:
                messages.info(request, 'Username OR password is incorrect')

        context = {}
        return render(request, 'accounts/login.html', context)

def logoutUser(request):
    logout(request)
    return redirect('login')

@login_required(login_url='login')
def home(request):
    return render(request, 'accounts/index.html')

@login_required(login_url='login')
def products(request):
	return render(request, 'accounts/products.html')

@login_required(login_url='login')
def productss(request):
    city=(request.POST.dict()['val1'])
    from geopy.geocoders import Nominatim
    import pickle
    geolocator = Nominatim(user_agent="my_user_agent")

    #city = "Pune"
    country = "INDIA"

    loc = geolocator.geocode(city + ',' + country)

    a = loc.latitude
    b = loc.longitude

    # # # #### load machine learning model
    loaded_model = pickle.load(open('knn_model.sav', 'rb'))
    predict1 = loaded_model.predict([[a, b]])
    print(predict1)
    context = {
        "p1": predict1[0]
    }
    return render(request, "accounts/shops.html", context)


from keras.models import load_model
import numpy as np
from keras.preprocessing import image

model = load_model("webapp/soil_weights.h5")

IMG_WIDTH = 224
IMG_HEIGHT = 224

X = [[3, 1], [1,1],[2,1],[0,1],[3,0],[1,0],[2,0],[0,0],[3,2],[1,2],[2,2],[0,2],[3,3],[1,3],[2,3],[0,3],[3,4],[1,4],[2,4],[0,4]]
y =['0', '1','2','3','4','5','6','7','8','9','10','11','12','13','14','15','16','17','18','19']

from sklearn.ensemble import RandomForestClassifier
clf=RandomForestClassifier(n_estimators=5)
clf.fit(X, y)

@login_required(login_url='login')
def predictImage(request):
    print(request.POST.dict(),'This is post')
    Region_Id=int(request.POST.dict()['region'])
    season = int(request.POST.dict()['region1'])
    fileObj = request.FILES["document"]
    fs = FileSystemStorage()
    filePathName = fs.save(fileObj.name, fileObj)
    filePathName = fs.url(filePathName)
    test_image = "." + filePathName
    img = image.load_img(test_image, target_size=(IMG_WIDTH, IMG_HEIGHT, 3))
    img = img_to_array(img)
    ph = (request.POST.dict()['val1'])
    b = float(request.POST.dict()['val2'])
    c = float(request.POST.dict()['val3'])
    d = float(request.POST.dict()['val4'])
    img = img / 255
    x = img.reshape(1, IMG_WIDTH, IMG_HEIGHT, 3)
    result = np.argmax(model.predict(x))
    


    text1=""
    if result==0:
        text1="Alluvial Image"
    elif result==1:
        text1="Black Soil Image"
    elif result==2:
        text1="Clay Soil Image"
    elif result==3:
        text1="Red Soil Image"

    text2=""
    if Region_Id==0:
        text2="Konkan"
    elif Region_Id==1:
        text2="Marathwada"
    elif Region_Id==2:
        text2="Vidarbha"
    elif Region_Id==3:
        text2="Pune"
    elif Region_Id==4:
        text2="Nashik"

    text3=""
    if season==0:
        text3="Kharip"
    elif season==1:
        text3="Rabi"
    elif season==2:
        text3="Zaib"

    loaded_model = pickle.load(open('rf_model.sav', 'rb'))
    locs = loaded_model.predict([[Region_Id, season,result,ph,b,c,d]])
    loaded_model1 = pickle.load(open('rf_model1.sav', 'rb'))
    locs1 = loaded_model1.predict([[Region_Id, season,result,ph,b,c,d]])
    humidity=locs1[0][0]
    temperature=locs1[0][1]
    rainfall=locs1[0][2]

    return render(request, "accounts/result.html",{"a1":text1,"a2":locs[0],"a3":humidity,"a4":temperature,"a5":rainfall,"a6":text2,"a7":text3,"a8":ph,"a9":b,"a10":c,"a11":d,'filePathName':filePathName})
