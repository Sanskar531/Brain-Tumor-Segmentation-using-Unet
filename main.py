import preprocessing as pp
import unet as un
import mat73
from scipy.io import loadmat
from sklearn.model_selection import train_test_split
import tensorflow.compat.v1 as tf

tf.enable_eager_execution(tf.ConfigProto(log_device_placement=True)) 
tf.config.set_soft_device_placement(False)

data = mat73.loadmat("./stacked_data.mat");
pp.preprocess_data(data);

data= loadmat("./N4Corrected2.mat"); 

X = data["X"]
y_mask = data["y_mask"]
y = data["y"]
X_train, X_test, y_train, y_test = train_test_split(X, y_mask, test_size=0.2)

model = un.unet((512,512,1))

print(model.summary());

CSV =  tf.compat.v1.keras.callbacks.CSVLogger("training_test", separator=',', append=True) 
history = model.fit(X_train, y_train, epochs=5, batch_size=8, steps_per_epoch=305, callbacks = [CSV]) 

model.save("./WeightedCross55epochs") 