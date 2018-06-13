#!/usr/bin/env python
import re
from sys import argv, stdout, stderr, exit, modules
from optparse import OptionParser

import tensorflow as tf
import numpy as np
import scipy

import math



# import ROOT with a fix to get batch mode (http://root.cern.ch/phpBB3/viewtopic.php?t=3198)
argv.append( '-b-' )
import ROOT
ROOT.gROOT.SetBatch(True)
ROOT.PyConfig.IgnoreCommandLineOptions = True
argv.remove( '-b-' )

from array import array

from HiggsAnalysis.CombinedLimit.DatacardParser import *
from HiggsAnalysis.CombinedLimit.ModelTools import *
from HiggsAnalysis.CombinedLimit.ShapeTools import *
from HiggsAnalysis.CombinedLimit.PhysicsModel import *
from HiggsAnalysis.CombinedLimit.tfscipyhess import ScipyTROptimizerInterface

parser = OptionParser(usage="usage: %prog [options] datacard.txt -o output \nrun with --help to get list of options")
addDatacardParserOptions(parser)
parser.add_option("-P", "--physics-model", dest="physModel", default="HiggsAnalysis.CombinedLimit.PhysicsModel:defaultModel",  type="string", help="Physics model to use. It should be in the form (module name):(object name)")
parser.add_option("--PO", "--physics-option", dest="physOpt", default=[],  type="string", action="append", help="Pass a given option to the physics model (can specify multiple times)")
parser.add_option("", "--dump-datacard", dest="dumpCard", default=False, action='store_true',  help="Print to screen the DataCard as a python config and exit")
parser.add_option("","--allowNegativeExpectation", default=False, action='store_true', help="allow negative expectation")
parser.add_option("","--freezePOIs", default=False, action='store_true', help="freeze POIs")
(options, args) = parser.parse_args()

if len(args) == 0:
    parser.print_usage()
    exit(1)

options.fileName = args[0]
if options.fileName.endswith(".gz"):
    import gzip
    file = gzip.open(options.fileName, "rb")
    options.fileName = options.fileName[:-3]
else:
    file = open(options.fileName, "r")

## Parse text file 
DC = parseCard(file, options)

if options.dumpCard:
    DC.print_structure()
    exit()

print(options)

nproc = len(DC.processes)
nsyst = len(DC.systs)
npoi = len(DC.signals)
if options.freezePOIs:
  npoi = 0

dtype = 'float64'

MB = ShapeBuilder(DC, options)

#determine number of bins for each channel
nbinschan = {}
nbinstotal = 0
for chan in DC.bins:
  expchan = DC.exp[chan]
  for proc in DC.processes:
    if proc in expchan:
      datahist = MB.getShape(chan,"data_obs")
      nbins = datahist.GetNbinsX()
      nbinschan[chan] = nbins
      nbinstotal += nbins
      break

#fill data, expected yields, and kappas

#n.b data and expected have shape [nbins]

#norm has shape [nbins,nprocs] and keeps track of expected normalization

#logkup/down have shape [nbins, nprocs, nsyst] and keep track of systematic variations
#per-bin, per-process, per nuisance-parameter 

logkepsilon = math.log(1e-3)

data_obs = np.empty([0],dtype=dtype)
norm = np.empty([0,nproc],dtype=dtype)
logkup = np.empty([0,nproc,nsyst],dtype=dtype)
logkdown = np.empty([0,nproc,nsyst],dtype=dtype)
for chan in DC.bins:
  expchan = DC.exp[chan]
  nbins = nbinschan[chan]
  
  datahist = MB.getShape(chan,"data_obs")
  datanp = np.array(datahist).astype(dtype)[1:-1]
  data_obs = np.concatenate((data_obs,datanp))
  
  normchan = np.empty([nbins,0],dtype=dtype)
  logkupchan = np.empty([nbins,0,nsyst],dtype=dtype)
  logkdownchan = np.empty([nbins,0,nsyst],dtype=dtype)
  for proc in DC.processes:
    hasproc = proc in DC.exp[chan]
    
    if hasproc:
      normhist = MB.getShape(chan,proc)
      normnp = np.array(normhist).astype(dtype)[1:-1]
      normnp = np.reshape(normnp,[-1,1])
      if not options.allowNegativeExpectation:
        normnp = np.maximum(normnp,0.)
    else:
      normnp = np.zeros([nbins,1],dtype=dtype)
      
    normchan = np.concatenate((normchan,normnp),axis=1)
      
    logkupproc = np.empty([nbins,1,0],dtype=dtype)
    logkdownproc = np.empty([nbins,1,0],dtype=dtype)
    for syst in DC.systs:
      name = syst[0]
      stype = syst[2]
      
      if not hasproc:
        logkupsyst = np.zeros([nbins,1,1],dtype=dtype)
        logkdownsyst = np.zeros([nbins,1,1],dtype=dtype)
      elif stype=='lnN':
        ksyst = syst[4][chan][proc]
        if type(ksyst) is list:
          ksystup = ksyst[1]
          ksystdown = ksyst[0]
          if ksystup == 0.:
            ksystup = 1.
          if ksystdown == 0.:
            ksystdown = 1.
          logkupsyst = math.log(ksystup)*np.ones([nbins,1,1],dtype=dtype)
          logkdownsyst = -math.log(ksystdown)*np.ones([nbins,1,1],dtype=dtype)
        else:
          if ksyst == 0.:
            ksyst = 1.
          logkupsyst = math.log(ksyst)*np.ones([nbins,1,1],dtype=dtype)
          logkdownsyst = math.log(ksyst)*np.ones([nbins,1,1],dtype=dtype)
        
      elif 'shape' in stype:
        kfac = syst[4][chan][proc]
        
        if kfac>0:
          normhistup = MB.getShape(chan,proc,name+"Up")
          normnpup = np.array(normhistup).astype(dtype)[1:-1]
          normnpup = np.reshape(normnpup,[-1,1])
          logkupsyst = kfac*np.log(normnpup/normnp)
          logkupsyst = np.where(np.equal(np.sign(normnp*normnpup),1), logkupsyst, logkepsilon*np.ones_like(logkupsyst))
          logkupsyst = np.reshape(logkupsyst,[-1,1,1])
          
          normhistdown = MB.getShape(chan,proc,name+"Down")
          normnpdown = np.array(normhistdown).astype(dtype)[1:-1]
          normnpdown = np.reshape(normnpdown,[-1,1])
          logkdownsyst = -kfac*np.log(normnpdown/normnp)
          logkdownsyst = np.where(np.equal(np.sign(normnp*normnpdown),1), logkdownsyst, -logkepsilon*np.ones_like(logkdownsyst))
          logkdownsyst = np.reshape(logkdownsyst,[-1,1,1])
        else:
          logkupsyst = np.zeros([normnp.shape[0],1,1],dtype=dtype)
          logkdownsyst = np.zeros([normnp.shape[0],1,1],dtype=dtype)
      else:
        raise Exception('Unsupported systematic type')

      logkupproc = np.concatenate((logkupproc,logkupsyst),axis=2)
      logkdownproc = np.concatenate((logkdownproc,logkdownsyst),axis=2)  

    logkupchan = np.concatenate((logkupchan,logkupproc),axis=1)
    logkdownchan = np.concatenate((logkdownchan,logkdownproc),axis=1)  
    
  norm = np.concatenate((norm,normchan), axis=0)

  logkup = np.concatenate((logkup,logkupchan),axis=0)
  logkdown = np.concatenate((logkdown,logkdownchan),axis=0)
  
  
print(np.max(np.abs(logkup)))
print(np.max(np.abs(logkdown)))
  
logkavg = 0.5*(logkup+logkdown)
logkhalfdiff = 0.5*(logkup-logkdown)

nexpnomv = np.sum(norm,axis=-1)

print("nbins = %d, ntotal = %e, npoi = %d, nsyst = %d" % (nexpnomv.shape[0], np.sum(nexpnomv), npoi, nsyst))

#list of signals preserving datacard order
signals = []
if not options.freezePOIs:
  for proc in DC.processes:
    if DC.isSignal[proc]:
      signals.append(proc)

systs = []
for syst in DC.systs:
  systs.append(syst[0])

#build matrix of signal strength effects
#hard-coded for now as one signal strength multiplier
#per signal process
logkr = np.zeros([nproc,npoi],dtype=dtype)
for ipoi,signal in enumerate(signals):
  iproc = DC.processes.index(signal)
  logkr[iproc][ipoi] = 1.

#initial value for signal strenghts
logrv = np.zeros([npoi]).astype(dtype)

#initial value for nuisances
thetav = np.zeros([nsyst]).astype(dtype)

#combined initializer for all fit parameters
logrthetav = np.concatenate((logrv,thetav),axis=0)


csignals = tf.constant(signals,name="csignals")
csysts = tf.constant(systs,name="csysts")

#data
#nobs = tf.placeholder(dtype, shape=data_obs.shape)
nobs = tf.Variable(data_obs, trainable=False, name="nobs")
theta0 = tf.Variable(np.zeros_like(thetav), trainable=False, name="theta0")
nexpnom = tf.Variable(nexpnomv, trainable=False, name="nexpnom")

#tf variable containing all fit parameters
logrtheta = tf.Variable(logrthetav, name="logrtheta")

#split back into signal strengths and nuisances
logr = logrtheta[:npoi]
theta = logrtheta[npoi:]

logr = tf.identity(logr,name="logr")
theta = tf.identity(theta,name="theta")

#matrices encoding effect of signal strengths
logrnorm = tf.reduce_sum(logkr*logr,axis=-1)

#interpolation for asymmetric log-normal
twox = 2.*theta
twox2 = twox*twox
alpha =  0.125 * twox * (twox2 * (3*twox2 - 10.) + 15.)
alpha = tf.clip_by_value(alpha,-1.,1.)
logk = logkavg + alpha*logkhalfdiff

#matrix encoding effect of nuisance parameters
logsnorm = tf.reduce_sum(logk*theta,axis=-1)

logrsnorm = logrnorm + logsnorm
rsnorm = tf.exp(logrsnorm)

#final expected yields per-bin including effect of signal
#strengths and nuisance parmeters
pnorm = rsnorm*norm
nexp = tf.reduce_sum(pnorm,axis=-1)
nexp = tf.identity(nexp,name='nexp')

nexpsafe = tf.where(tf.equal(nobs,tf.zeros_like(nobs)), tf.ones_like(nobs), nexp)
lognexp = tf.log(nexpsafe)

nexpnomsafe = tf.where(tf.equal(nobs,tf.zeros_like(nobs)), tf.ones_like(nobs), nexpnom)
lognexpnom = tf.log(nexpnomsafe)


#final likelihood computation

#poisson term
lnfull = tf.reduce_sum(-nobs*lognexp + nexp, axis=-1)

#poisson term with offset to improve numerical precision
ln = tf.reduce_sum(-nobs*(lognexp-lognexpnom) + nexp-nexpnom, axis=-1)

#constraints
lc = tf.reduce_sum(0.5*tf.square(theta - theta0))

l = ln + lc
l = tf.identity(l,name="loss")

lfull = lnfull + lc
lfull = tf.identity(lfull,name="lossfull")

basename = '.'.join(options.fileName.split('.')[:-1])
tf.train.export_meta_graph(filename='%s.meta' % basename)
