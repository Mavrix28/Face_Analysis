from django.shortcuts import render
from django.http import HttpResponse
from app.forms import FaceRecognitionform
from app.machinelearning import pipeline_model
from django.conf import settings
from app.models import FaceRecognition
import os  
# Create your views here.


def index(request):
    model = FaceRecognitionform()
    if request.method == 'POST':
        model = FaceRecognitionform(request.POST or None, request.FILES or None)
        if model.is_valid():
                # Save the form and retrieve the saved instance
                saved_instance = model.save()

                # Extract the primary key from the saved instance
                primary_key = saved_instance.id

                # Retrieve the saved model instance from the database (optional)
                imgobj = FaceRecognition.objects.get(pk=primary_key)

                # Get the file path of the uploaded image
                fileroot = str(imgobj.image)
                filepath = os.path.join(settings.MEDIA_ROOT, fileroot)

                # Process the image using the pipeline model
                results = pipeline_model(filepath)
                print(results)
                
                return render(request, 'index.html', {'form': model,'upload':True,'results':results})
        else:
                 print("The form is not valid.")
                 
    
    return render(request, 'index.html', {'form': model,'upload':False})