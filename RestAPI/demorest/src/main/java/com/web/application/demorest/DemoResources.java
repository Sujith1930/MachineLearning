package com.web.application.demorest;

import java.util.ArrayList;
import java.util.List;

import com.web.application.demorest.DemoPeople;

import jakarta.ws.rs.GET;
import jakarta.ws.rs.Path;
import jakarta.ws.rs.PathParam;
import jakarta.ws.rs.Produces;
import jakarta.ws.rs.core.MediaType;

@Path("bpdts-test-app")
public class DemoResources {
	
	DemoRepository repo = new DemoRepository();
	
	@GET
	@Produces({MediaType.APPLICATION_JSON})
	
	public List<DemoPeople> getDemoPeople() {
		
		System.out.println("Demo people called");
		return repo.getDemoPeople();
	}
	
	@GET
	@Path("/{location}")
	@Produces({MediaType.APPLICATION_JSON})
	public DemoPeople getDemo(@PathParam("location") String location) {
		return repo.getDemoPeople(location);
	}
}
