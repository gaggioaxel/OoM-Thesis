import numpy 
import scipy.stats

def extractFeatures(raw_feat):
  mean = numpy.mean(raw_feat, axis=0)
  var = numpy.var(raw_feat, axis=0)
  kurt = scipy.stats.kurtosis(raw_feat, axis=0, bias=True)
  skew = scipy.statsskew(raw_feat, axis=0, bias=True)
  corr = numpy.corrcoef(raw_feat, rowvar=False)[numpy.triu_indices(raw_feat,k=1)]
  mad = numpy.mean(numpy.abs(raw_feat - mean), axis=0)
  sem = numpy.std(raw_feat, axis=0) / numpy.sqrt(raw_feat.shape[0])
  energy = numpy.sqrt(numpy.sum(raw_feat ** 2, axis=0))
  iqr = scipy.stats.iqr(raw_feat,axis=0)
  mi = numpy.min(raw_feat, axis=0)
  ma = numpy.max(raw_feat, axis=0)

  fft = numpy.fft.fft(raw_feat, axis=0)
  amplitude_spectrum = numpy.abs(fft)
  frq_info = [numpy.angle(fft)[0, :],        # First freq
              numpy.mean(fft.real, axis=0),  
              numpy.max(fft.real, axis=0),   
              numpy.argmax(fft.real, axis=0),
              numpy.min(fft.real, axis=0),   
              numpy.argmin(fft.real, axis=0),
              scipy.stats.skew(amplitude_spectrum, axis=0, bias=True),
              scipy.stats.kurtosis(amplitude_spectrum, axis=0, bias=True) ] 
  
  refined_features = numpy.hstack([mean,var,kurt,skew,corr,mad,sem,energy,iqr,mi,ma,numpy.hstack(frq_info)])
  
  return refined_features


from imblearn.over_sampling import BorderlineSMOTE
from sklearn.ensemble import RandomForestClassifier

# Define a function to perform the model training and prediction for one iteration
def train_predict(X, y, i):
    
    # Remove i-th sample from training 
    X_tr = numpy.delete(X, i, axis=0)
    y_tr = numpy.delete(y, i, axis=0)

    # Count labels occurencies
    labels_count = numpy.bincount(y_tr)
    ratio = 0.4

    # Defines the neighborhood of samples to use to generate the synthetic samples.
    k_neigh = numpy.clip(int(labels_count[1] * ratio), 1, len(y))

    # Determine if a minority sample is in "danger"
    m_neigh = numpy.clip(int(labels_count[0] * ratio), 1, len(y)) 
    
    # Fit the model with different hyperparameters using BorderlineSMOTE
    bsmote = BorderlineSMOTE(k_neigh, m_neigh)
    X_tr_resampled, y_tr_resampled = bsmote.fit_resample(X_tr, y_tr)

    # Train a first forest
    rf = RandomForestClassifier(n_estimators=500)
    rf.fit(X_tr_resampled,y_tr_resampled)

    # Sort importances in descending order and select the top 50 features
    top_features = numpy.argsort(numpy.array(rf.feature_importances_))[::-1][:50]

    # Select the most important features on train and test set
    X_tr_star_resampled = X_tr_resampled[:, top_features]
    X_t_star_resampled = X[i, top_features].reshape(1, -1)

    # Train a new forest
    rf = RandomForestClassifier(n_estimators=200, max_features=None)
    rf.fit(X_tr_star_resampled, y_tr_resampled)

    # Return prediction on the test instance
    return rf.predict(X_t_star_resampled)[0]

# LOOCV on a dataset
X = ... ; y = ... 
predicted_labels = [train_predict(X, y, i) for i in range(len(y))]