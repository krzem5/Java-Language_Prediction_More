package com.krzem.language_prediction_more;



import com.krzem.NN.NeuralNetwork;
import java.io.File;
import java.lang.Math;
import java.nio.file.Files;
import java.nio.file.Paths;
import java.util.Arrays;
import java.util.Collections;
import java.util.List;



class LangPrediction{
	public static final String[] LANGS={"de","en","es","fi","fr","it","nl","pl","sl","sv"};
	public static final String ALPHABET="abcdefghijklmnopqrstuvwxyz";
	public static final int MAX_LETTERS=40;
	public static final int DATASET_SIZE=1000;
	public static String[][] WORDS;
	public NeuralNetwork NN;
	public String fp;



	public LangPrediction(String fp){
		this.fp=fp;
		if (new File(this.fp+".nn-data").exists()){
			this.NN=NeuralNetwork.fromFile(this.fp);
		}
		else{
			this.NN=new NeuralNetwork(MAX_LETTERS*ALPHABET.length(),new int[]{MAX_LETTERS*2},LANGS.length,5.5e-3);
		}
		this._load_words();
	}



	public double[] encode_word(String w){
		w=w.toLowerCase();
		double[] e=new double[MAX_LETTERS*ALPHABET.length()];
		int i=0;
		for (int j=0;j<w.length();j++){
			if (j>=MAX_LETTERS||ALPHABET.indexOf(w.charAt(j))==-1){
				break;
			}
			e[i*ALPHABET.length()+ALPHABET.indexOf(w.charAt(j))]=1;
			i++;
		}
		return e;
	}



	public void predict(String W){
		double[] o=this.NN.predict(this.encode_word(W));
		double s=0;
		for (double v:o){
			s+=v;
		}
		int i=0;
		for (double v:o){
			o[i]=v/s;
			i++;
		}
		i=0;
		for (String k:this.LANGS){
			System.out.println(String.format("%s: %s -> %f",W,k,o[i]));
			i++;
		}
	}



	public void predict_sentence(String S){
		double[] o=new double[this.LANGS.length];
		int j=0;
		for (String W:S.split(" ")){
			double[] p=this.NN.predict(this.encode_word(W));
			double s=0;
			for (double v:p){
				s+=v;
			}
			int i=0;
			for (double v:p){
				o[i]+=v/s;
				i++;
			}
			j++;
		}
		int i=0;
		for (double v:o){
			o[i]=v/j;
			i++;
		}
		i=0;
		for (String k:this.LANGS){
			System.out.println(String.format("%s: %s -> %f",S,k,o[i]));
			i++;
		}
	}



	public void test(){
		this.NN.test(this._batch(this.DATASET_SIZE));
	}



	public void train(int itr,int s,int l_steps,double ea){
		int p=0;
		double[][][] td=this._test_batch();
		double tm=System.nanoTime();
		for (int i=s+0;i<itr;i++){
			double[][][] d=this._batch(this.DATASET_SIZE);
			if (Math.floor((double)(i)/itr*l_steps)>p){
				double df=System.nanoTime()-tm;
				tm=System.nanoTime();
				p=(int)Math.floor((double)(i)/itr*l_steps);
				String n=Double.toString((double)(p)/(l_steps/100d)).replace("\\.0$","");
				double acc=this.NN.acc(td);
				System.out.println(String.format("%s %% complete, Time=%f, Acc=%f",n,df*1e-9,acc));
				this.NN.toFile(String.format("%s-%d",this.fp,i));
				if (acc>=ea){
					break;
				}
			}
			for (int k=0;k<d.length;k++){
				this.NN.train(d[k][0],d[k][1]);
			}
		}
		this.NN.toFile(this.fp+"-full");
	}



	public double acc(){
		return this.NN.acc(this._batch(this.DATASET_SIZE*this.LANGS.length));
	}



	private void _load_words(){
		try{
			this.WORDS=new String[this.LANGS.length][];
			int i=0;
			for (String l:this.LANGS){
				this.WORDS[i]=new String(Files.readAllBytes(Paths.get(String.format("./data/%s.txt",l))),"UTF-8").split("\n");
				i++;
			}
		}
		catch (Exception e){
			e.printStackTrace();
		}
	}



	private double[] encode_lang(int i){
		double[] o=new double[this.LANGS.length];
		o[i]=1;
		return o;
	}



	private double[][][] _test_batch(){
		return this._batch(this.DATASET_SIZE/2);
	}



	private double[][][] _batch(int sz){
		double[][][] d=new double[sz][][];
		int i=0;
		int j=0;
		for (String l:this.LANGS){
			List<String> tl=Arrays.asList(this.WORDS[i]);
			Collections.shuffle(tl);
			String[] rw=new String[sz/this.LANGS.length];
			for (int k=0;k<sz/this.LANGS.length;k++){
				rw[k]=tl.get(k);
			}
			for (String w:rw){
				double[][] dt={this.encode_word(w),this.encode_lang(i)};
				d[j]=dt;
				j++;
			}
			i++;
		}
		return d;
	}
}
