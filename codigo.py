import speech_recognition
import librosa
import math
import dtw as dt
import numpy as np
import os

def dist(v1, v2):
	s = 0
	for i in range(len(v1)):
		s+=(v1[i]-v2[i])**2
	return math.sqrt(s)

def add_command(fname):
	r = speech_recognition.Recognizer()

	with speech_recognition.Microphone() as source:
		# r.adjust_for_ambient_noise(source)
		print("Ouvindo...")
		audio  = r.listen(source)
		print("Gravado.")
		b = audio.get_aiff_data()
		f = open(fname, "wb")
		f.write(b)
		f.close()
		y, sr = librosa.load(fname)
		mfcc = librosa.feature.mfcc(y, sr)
		f = open(fname, "w")
		for x in mfcc:
			for z in x:
				f.write(str(z)+" ")
			f.write("\n")

def listen_command():
	r = speech_recognition.Recognizer()

	with speech_recognition.Microphone() as source:
		# r.adjust_for_ambient_noise(source)
		print("Esperando comando de voz...")
		audio  = r.listen(source)
		print("Comando recebido.")
		b = audio.get_aiff_data()
		f = open("comm.aif", "wb")
		f.write(b)
		f.close()
		y, sr = librosa.load("comm.aif")
		mfcc = librosa.feature.mfcc(y, sr)
		return mfcc

mfnames = []
mfccs = []
escolher = []
inserir  = []
encerrar = []

def initialize_system():
	global mfnames
	global mfccs
	global escolher
	global inserir
	global encerrar
	f = open("musics.txt", "r")
	l = f.readlines()
	f.close()
	mfnames = [x[:-1] for x in l]

	for x in mfnames:
		f = open(x+".txt", "r")
		l = f.readlines()
		f.close()
		temp = [y.split() for y in l]
		temp = np.array(temp)
		mfccs.append(temp.astype(np.float64))

	f = open("escolher.txt", "r")
	lines = f.readlines()
	f.close()
	escolher = [x.split() for x in lines]
	escolher = np.array(escolher)
	escolher = escolher.astype(np.float64)

	f = open("inserir.txt", "r")
	lines = f.readlines()
	f.close()
	inserir = [x.split() for x in lines]
	inserir = np.array(inserir)
	inserir = inserir.astype(np.float64)

	f = open("encerrar.txt", "r")
	lines = f.readlines()
	f.close()
	encerrar = [x.split() for x in lines]
	encerrar = np.array(encerrar)
	encerrar = encerrar.astype(np.float64)

def insert_music():
	global mfnames
	global mfccs
	mfname = input("Digite o nome do arquivo da música: ")
	mfnames.append(mfname)
	f = open("musics.txt", "a")
	f.write(mfname+"\n")
	f.close()
	r = speech_recognition.Recognizer()

	with speech_recognition.Microphone() as source:
		# r.adjust_for_ambient_noise(source)
		print("Fale o nome da música...")
		audio  = r.listen(source)
		print("Gravado.")
		b = audio.get_aiff_data()
		f = open("comm.aif", "wb")
		f.write(b)
		f.close()
		y, sr = librosa.load("comm.aif")
		mfcc = librosa.feature.mfcc(y, sr)
		mfccs.append(mfcc)
		f = open(mfname+".txt", "w")
		for x in mfcc:
			for z in x:
				f.write(str(z)+" ")
			f.write("\n")

def choose_music():
	global mfnames
	global mfccs
	r = speech_recognition.Recognizer()

	with speech_recognition.Microphone() as source:
		# r.adjust_for_ambient_noise(source)
		print("Escolha uma música...")
		audio  = r.listen(source)
		print("Gravado.")
		b = audio.get_aiff_data()
		f = open("comm.aif", "wb")
		f.write(b)
		f.close()
		y, sr = librosa.load("comm.aif")
		mfcc = librosa.feature.mfcc(y, sr)

	dists = []
	for x in mfccs:
		dists.append(dt.dtw(x.T, mfcc.T, dist=lambda x, y: dist(x,y))[0])
	m = min(dists)
	for i in range(len(mfnames)):
		if dists[i]==m:
			print ("Abrindo arquivo de música %s" % mfnames[i])
			# os.system("start wmplayer.exe %s" % mfnames[i])

def main():
	initialize_system()
	while True:
		mfcc = listen_command()
		dists = []
		dists.append(dt.dtw(escolher.T, mfcc.T, dist=lambda x, y: dist(x,y))[0])
		dists.append(dt.dtw(inserir.T, mfcc.T, dist=lambda x, y: dist(x,y))[0])
		dists.append(dt.dtw(encerrar.T, mfcc.T, dist=lambda x, y: dist(x,y))[0])
		m = min(dists)
		if (dists[0]==m): 
			choose_music()
		elif (dists[1]==m): 
			insert_music()
		elif (dists[2]==m):
			print("Encerrando...")
			break

if __name__=="__main__":
	main()
	# add_command("inerir.txt")
