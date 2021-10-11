package com.web.application.demorest;

import java.util.Collection;
import java.util.Iterator;
import java.util.List;
import java.util.ListIterator;

@jakarta.xml.bind.annotation.XmlRootElement
public class DemoPeople  {
	
	private String name;
	private String location;
	private double miles;
	
	public String getName() {
		return name;
	}
	public void setName(String name) {
		this.name = name;
	}
	public String getLocation() {
		return location;
	}
	public void setLocation(String location) {
		this.location = location;
	}
	public double getMiles() {
		return miles;
	}
	public void setMiles(double miles) {
		this.miles = miles;
	}
	}

