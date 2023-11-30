import numpy 
import scipy.stats

def extractFeatures(features):
  raw_features = features.shape[1]
  arr = features
  energy = numpy.sqrt(numpy.sum(arr ** 2, axis=0))
  fft = numpy.fft.fft(arr, axis=0)
  amplitude_spectrum = numpy.abs(fft)
  phase_angle = numpy.angle(fft) 

  frq_info = [
              phase_angle[0, :],
              numpy.mean(fft.real, axis=0),
              numpy.max(fft.real, axis=0),
              numpy.argmax(fft.real, axis=0),
              numpy.min(fft.real, axis=0),
              numpy.argmin(fft.real, axis=0),
              scipy.stats.skew(amplitude_spectrum, axis=0, bias=True),
              scipy.stats.kurtosis(amplitude_spectrum, axis=0, bias=True),
  ] 

  frq_info = numpy.hstack(frq_info)
  mean = numpy.mean(arr, axis=0)
  var = numpy.var(arr, axis=0)
  kurt = scipy.stats.kurtosis(arr, axis=0, bias=True)
  skew_ = scipy.statsskew(arr, axis=0, bias=True)
  corr = numpy.corrcoef(arr, rowvar=False)[numpy.triu_indices(raw_features,k=1)]
  mad = numpy.mean(numpy.abs(arr - mean), axis=0)
  sem = numpy.std(arr, axis=0) / numpy.sqrt(arr.shape[0])
  mi = numpy.min(arr, axis=0)
  ma = numpy.max(arr, axis=0)
  return numpy.hstack([mean,var,kurt,skew_,corr,mad,sem,energy,scipy.stats.iqr(arr,axis=0),mi,ma,frq_info])
