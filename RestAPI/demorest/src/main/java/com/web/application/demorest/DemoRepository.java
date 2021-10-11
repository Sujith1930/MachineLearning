package com.web.application.demorest;

import java.util.ArrayList;
import java.util.List;

public class DemoRepository {

	List<DemoPeople> dmp;

	public DemoRepository() {

		dmp = new ArrayList<>();
		DemoPeople dp = new DemoPeople();
		dp.setName("Sam");
		dp.setLocation("Barnet");
		dp.setMiles(12.3);

		DemoPeople dp1 = new DemoPeople();
		dp1.setName("Samuel");
		dp1.setLocation("Bellingham");
		dp1.setMiles(8.1);

		DemoPeople dp2 = new DemoPeople();
		dp2.setName("Tim");
		dp2.setLocation("Chelsea");
		dp2.setMiles(2.5);

		DemoPeople dp3 = new DemoPeople();
		dp3.setName("Lucy");
		dp3.setLocation("Croydon");
		dp3.setMiles(10.1);

		DemoPeople dp4 = new DemoPeople();
		dp4.setName("Tymon");
		dp4.setLocation("Edgeware");
		dp4.setMiles(10.8);

		DemoPeople dp5 = new DemoPeople();
		dp5.setName("Sid");
		dp5.setLocation("Enfield");
		dp5.setMiles(11.3);

		DemoPeople dp6 = new DemoPeople();
		dp6.setName("Claire");
		dp6.setLocation("Birmingham");
		dp6.setMiles(128);

		DemoPeople dp7 = new DemoPeople();
		dp7.setName("Vanessa");
		dp7.setLocation("Barnet");
		dp7.setMiles(12.3);

		DemoPeople dp8 = new DemoPeople();
		dp8.setName("Lauren");
		dp8.setLocation("Harrow");
		dp8.setMiles(13.8);

		DemoPeople dp9 = new DemoPeople();
		dp9.setName("Carol");
		dp9.setLocation("Wales");
		dp9.setMiles(204);

		dmp.add(dp);
		dmp.add(dp1);
		dmp.add(dp2);
		dmp.add(dp3);
		dmp.add(dp4);
		dmp.add(dp5);
		dmp.add(dp6);
		dmp.add(dp7);
		dmp.add(dp8);
		dmp.add(dp9);
	}

	public List<DemoPeople> getDemoPeople() {
		
		List<DemoPeople> dpr = new ArrayList<>();

		for (DemoPeople d : dmp) {

			if (d.getMiles() <= 50.0 || d.getLocation() == "London") {

				dpr.add(d);
			}
		}
		return dpr;

	}

	public DemoPeople getDemoPeople(String location) {
		for (DemoPeople d : dmp) {

			if (d.getLocation() == location) {
				return d;
			}
		}

		return null;

	}

}
