from django import forms
from app.models import FaceRecognition

class FaceRecognitionform(forms.ModelForm):
    class Meta:
        model = FaceRecognition
        fields = ['image']
      
    def __init__(self, *args, **kwargs):
        super(FaceRecognitionform, self).__init__(*args, **kwargs)  
        self.fields['image'].widget.attrs.update({'class': 'form-control'})
        