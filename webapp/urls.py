from django.urls import path
from . import views

urlpatterns = [
	path('register/', views.registerPage, name="register"),
	path('login/', views.loginPage, name="login"),
	path('logout/', views.logoutUser, name="logout"),
	path('', views.home, name="home"),
	path("predict",views.predictImage,name='predictImage'),
	path('products/', views.products, name='products'),
	path("industry", views.productss, name='productss'),

]
