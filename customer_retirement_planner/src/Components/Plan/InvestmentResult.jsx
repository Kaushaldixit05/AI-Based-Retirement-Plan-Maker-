import React from 'react';

const InvestmentResult = ({ formData, investmentStrategy }) => {
  return (
    <div className="investment-strategy">
      <h4>Investment Strategy</h4>
      <table className="table table-bordered">
        <thead>
          <tr>
            <th>Input</th>
            <th>Value</th>
          </tr>
        </thead>
        <tbody>
          {/* Display input values */}
          <tr>
            <td>Current Age</td>
            <td>{formData.currentAge}</td>
          </tr>
          <tr>
            <td>Retirement Age</td>
            <td>{formData.retirementAge}</td>
          </tr>
          <tr>
            <td>Desired Retirement Fund</td>
            <td>{formData.desiredFund}</td>
          </tr>
          <tr>
            <td>Monthly Investment</td>
            <td>{formData.monthlyInvestment}</td>
          </tr>
          <tr>
            <td>Risk Category</td>
            <td>{formData.riskCategory}</td>
          </tr>
        </tbody>
      </table>
      {/* Display investment suggestions */}
      <h4>Investment Suggestions</h4>
      <table className="table table-bordered">
        <thead>
          <tr>
            <th>Stock Name</th>
            <th>Annual Return (%)</th>
            <th>Future Value (INR)</th>
            <th>Investment Percentage</th>
            <th>Risk Profile</th>
          </tr>
        </thead>
        <tbody>
          {investmentStrategy['Investment Suggestions'].map((suggestion, index) => (
            <tr key={index}>
              <td>{suggestion['Stock Name']}</td>
              <td>{suggestion['Annual Return (%)']}</td>
              <td>{suggestion['Future Value (INR)']}</td>
              <td>{suggestion['Investment Percentage']}</td>
              <td>{suggestion['Risk Profile']}</td>
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  );
};

export default InvestmentResult;