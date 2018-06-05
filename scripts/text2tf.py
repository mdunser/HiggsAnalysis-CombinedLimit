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
parser.add_option("-t","--toys", default=0, type=int, help="run a given number of toys, 0 fits the data (default), and -1 fits the asimov toy")
parser.add_option("","--toysFrequentist", default=True, action='store_true', help="run frequentist-type toys by randomizing constraint minima")
parser.add_option("","--bypassFrequentistFit", default=True, action='store_true', help="bypass fit to data when running frequentist toys to get toys based on prefit expectations")
parser.add_option("","--bootstrapData", default=False, action='store_true', help="throw toys directly from observed data counts rather than expectation from templates")
parser.add_option("","--randomizeStart", default=False, action='store_true', help="randomize starting values for fit (only implemented for asimov dataset for now")
parser.add_option("","--tolerance", default=1e-3, type=float, help="convergence tolerance for minimizer")
parser.add_option("","--expectSignal", default=1., type=float, help="rate multiplier for signal expectation (used for fit starting values and for toys)")
parser.add_option("","--seed", default=123456789, type=int, help="random seed for toys")
(options, args) = parser.parse_args()

if len(args) == 0:
    parser.print_usage()
    exit(1)
    
seed = options.seed
print(seed)
np.random.seed(seed)
tf.set_random_seed(seed)

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

dtype = 'float64'
#dtype = 'float32'

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
          if np.any(np.logical_and(np.equal(normnpup,0.),np.not_equal(normnp,0.))):
            print(['up',chan,proc,name])
          logkupsyst = np.where(np.equal(normnpup,0.),logkepsilon*np.ones_like(logkupsyst),logkupsyst)
          #logkupsyst = np.where(np.equal(normnpup,0.),np.zeros_like(logkupsyst),logkupsyst)
          logkupsyst = np.where(np.equal(normnp,0.),np.zeros_like(logkupsyst),logkupsyst)
          logkupsyst = np.reshape(logkupsyst,[-1,1,1])
          
          normhistdown = MB.getShape(chan,proc,name+"Down")
          normnpdown = np.array(normhistdown).astype(dtype)[1:-1]
          normnpdown = np.reshape(normnpdown,[-1,1])
          logkdownsyst = -kfac*np.log(normnpdown/normnp)
          if np.any(np.logical_and(np.equal(normnpdown,0.),np.not_equal(normnp,0.))):
            print(['down',chan,proc,name])
          logkdownsyst = np.where(np.equal(normnpdown,0.),-logkepsilon*np.ones_like(logkdownsyst),logkdownsyst)
          #logkdownsyst = np.where(np.equal(normnpdown,0.),np.zeros_like(logkdownsyst),logkdownsyst)
          logkdownsyst = np.where(np.equal(normnp,0.),np.zeros_like(logkdownsyst),logkdownsyst)
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

#list of signals preserving datacard order
signals = []
for proc in DC.processes:
  if DC.isSignal[proc]:
    signals.append(proc)

#build matrix of signal strength effects
#hard-coded for now as one signal strength multiplier
#per signal process
logkr = np.zeros([nproc,npoi],dtype=dtype)
for ipoi,signal in enumerate(signals):
  iproc = DC.processes.index(signal)
  logkr[iproc][ipoi] = 1.

#initial value for signal strenghts
#rv = options.expectSignal*np.ones([npoi]).astype(dtype)
logrv = math.log(options.expectSignal) + np.zeros([npoi]).astype(dtype)


#initial value for nuisances
thetav = np.zeros([nsyst]).astype(dtype)

#combined initializer for all fit parameters
logrthetav = np.concatenate((logrv,thetav),axis=0)


#data
#nobs = tf.placeholder(dtype, shape=data_obs.shape)
nobs = tf.Variable(data_obs, trainable=False)
theta0 = tf.Variable(np.zeros_like(thetav), trainable=False)
nexpnom = tf.Variable(nexpnomv, trainable=False)

#tf variable containing all fit parameters
logrtheta = tf.Variable(logrthetav)

#split back into signal strengths and nuisances
logr = logrtheta[:npoi]
theta = logrtheta[npoi:]

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

grads = tf.gradients(l,logrtheta)
grads = tf.identity(grads,"loss_grads")

grad = grads[0]

#uncertainty computation
hess = tf.hessians(l,logrtheta)[0]
hess = tf.identity(hess,name="loss_hessian")

#initialize output tree
f = ROOT.TFile( 'fitresults_%i.root' % seed, 'recreate' )
tree = ROOT.TTree("fitresults", "fitresults")

tseed = array('i', [seed])
tree.Branch('seed',tseed,'seed/I')

titoy = array('i',[0])
tree.Branch('itoy',titoy,'itoy/I')

tstatus = array('i',[0])
tree.Branch('status',tstatus,'status/I')

terrstatus = array('i',[0])
tree.Branch('errstatus',terrstatus,'errstatus/I')

tedmval = array('f',[0])
tree.Branch('edmval',tedmval,'edmval/F')

tsigvals = []
tsigerrs = []
for sig in signals:
  tsigval = array('f', [0.])
  tsigerr = array('f', [0.])
  tsigvals.append(tsigval)
  tsigerrs.append(tsigerr)
  tree.Branch(sig, tsigval, '%s/F' % sig)
  tree.Branch('%s_err' % sig, tsigerr, '%s_err/F' % sig)

tthetavals = []
ttheta0vals = []
tthetaerrs = []
for syst in DC.systs:
  systname = syst[0]
  tthetaval = array('f', [0.])
  ttheta0val = array('f', [0.])
  tthetaerr = array('f', [0.])
  tthetavals.append(tthetaval)
  ttheta0vals.append(ttheta0val)
  tthetaerrs.append(tthetaerr)
  tree.Branch(systname, tthetaval, '%s/F' % systname)
  tree.Branch('%s_In' % systname, ttheta0val, '%s_In/F' % systname)
  tree.Branch('%s_err' % systname, tthetaerr, '%s_err/F' % systname)

#initialize tf session
sess = tf.Session()
sess.run(tf.global_variables_initializer())

print("min norm = %e" % np.amin(norm))
print("min nexp = %e" % np.amin(sess.run(nexp)))

ntoys = options.toys
if ntoys <= 0:
  ntoys = 1

xtol = np.finfo(dtype).eps

#set likelihood offset
sess.run(nexpnom.assign(nexp))

#prefit to data if needed
if options.toys>0 and options.toysFrequentist and not options.bypassFrequentistFit:  
  minimizer = ScipyTROptimizerInterface(l, options={'verbose': 3, 'maxiter' : 100000, 'gtol' : 0., 'xtol' : xtol, 'barrier_tol' : 0.})
  ret = minimizer.minimize(sess)
  logrthetav = sess.run(thetav)

def printfunc(l):
  print(l)

def printstep(x,y):
  print([x,y])

for itoy in range(ntoys):
  titoy[0] = itoy
  
  sess.run(logrtheta.assign(logrthetav))
    
  dofit = True
  
  if options.toys < 0:
    print("Running fit to asimov toy")
    sess.run(nobs.assign(nexp))
    if options.randomizeStart:
      hessval = sess.run(hess)
      invhess = np.linalg.inv(hessval)
      sess.run(logrtheta.assign(np.random.multivariate_normal(logrthetav,invhess)))
    else:
      dofit = False
  elif options.toys == 0:
    print("Running fit to observed data")
    sess.run(nobs.assign(data_obs))
  else:
    print("Running toy %i" % itoy)
    if options.bootstrapData:
      #randomize from observed data
      sess.run(nobs.assign(tf.random_poisson(nobs,shape=[],dtype=dtype)))
    else:
      #randomize from expectation
      sess.run(nobs.assign(tf.random_poisson(nexp,shape=[],dtype=dtype)))  
  
    if options.toysFrequentist:
      #randomize nuisance constraint minima
      sess.run(theta0.assign(theta + tf.random_normal(shape=thetav.shape,dtype=dtype)))
      thetatmp = sess.run(theta0)
      logrthetatmp = np.concatenate((logrthetav[:npoi],thetatmp),axis=0)
      sess.run(logrtheta.assign(logrthetatmp))
    else:
      #randomize actual values
      sess.run(logrtheta.assign(tf.concat([logr,theta+tf.random_normal(shape=thetav.shape,dtype=dtype)],axis=0)))

  if dofit:
    #set likelihood offset
    sess.run(nexpnom.assign(nexp))
    minimizer = ScipyTROptimizerInterface(l, options={'verbose': 3, 'maxiter' : 100000, 'gtol' : 0., 'xtol' : xtol, 'barrier_tol' : 0.})
    ret = minimizer.minimize(sess)

  xval, fval, gradval, hessval = sess.run([logrtheta,l,grad,hess])

  isposdef = np.all(np.greater_equal(np.linalg.eigvalsh(hessval),0.))
    
  #get fit values
  logrvals = xval[:npoi]
  thetavals = xval[npoi:]  
  
  #transformation from logr to r
  rvals = np.exp(logrvals)
  jac = np.diagflat(np.concatenate((rvals,np.ones_like(thetav)),axis=0))
  jact = np.transpose(jac)
    
  try:        
    invhess = np.linalg.inv(hessval)
    edmval = 0.5*np.matmul(np.matmul(np.transpose(gradval),invhess),gradval)
    
    #transformation from logr to r for covariance matrix
    invhess = np.matmul(jac,np.matmul(invhess,jact))
    sigmasv = np.sqrt(np.diag(invhess))
    errstatus = 0
    if np.any(np.isnan(sigmasv)):
      errstatus = 1
  except np.linalg.LinAlgError:
    edmval = -99.
    sigmasv = -99.*np.ones_like(xval)
    errstatus = 2
  
  if isposdef and edmval > 0.:
    status = 0
  else:
    status=1
    
  print("status = %i, errstatus = %i, fval = %f, edmval = %e" % (status,errstatus,fval,edmval))
  
  tstatus[0] = status
  terrstatus[0] = errstatus
  tedmval[0] = edmval
  
  rsigmasv = sigmasv[:npoi]
  thetasigmasv = sigmasv[npoi:]
  
  theta0vals = sess.run(theta0)

  for sig,sigval,sigma,tsigval,tsigerr in zip(signals,rvals,rsigmasv,tsigvals,tsigerrs):
    tsigval[0] = sigval
    tsigerr[0] = sigma
    if itoy==0:
      print('%s = %f +- %f' % (sig,sigval,sigma))

  for syst,thetaval,theta0val,sigma,tthetaval,ttheta0val,tthetaerr in zip(DC.systs,thetavals,theta0vals,thetasigmasv,tthetavals,ttheta0vals,tthetaerrs):
    tthetaval[0] = thetaval
    ttheta0val[0] = theta0val
    tthetaerr[0] = sigma
    if itoy==0:
      print('%s = %f +- %f (%s_In = %f)' % (syst[0], thetaval, sigma, syst[0],theta0val))
    
  tree.Fill()

f.Write()
f.Close()
