package com.krzem.language_prediction_more;



public class Main{
	public static void main(String[] args){
		new Main();
	}



	public Main(){
		LangPrediction l=new LangPrediction("./nn/data");
		System.out.println(l.acc());
		l.train(100000,2996,10000,95);
		System.out.println(l.acc());
	}
}